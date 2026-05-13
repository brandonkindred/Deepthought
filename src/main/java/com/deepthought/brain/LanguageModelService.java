package com.qanairy.brain;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import com.deepthought.models.GraphLanguageModel;
import com.deepthought.models.Token;
import com.deepthought.models.edges.TokenWeight;
import com.deepthought.models.repository.GraphLanguageModelRepository;
import com.deepthought.models.repository.TokenWeightRepository;

/**
 * Materializes a bigram language model from the current state of the Neo4j knowledge graph.
 *  Every {@code HAS_RELATED_TOKEN} edge is treated as one observation of a (source, target)
 *  transition weighted by its learned Q-value, and each source row is normalized into a
 *  probability distribution over next tokens. The resulting transition table is persisted
 *  as a {@link GraphLanguageModel} node so subsequent generate calls reference it by id
 *  rather than re-walking the graph.
 */
@Component
public class LanguageModelService {
	private static final Logger log = LoggerFactory.getLogger(LanguageModelService.class);

	private final TokenWeightRepository token_weight_repo;
	private final GraphLanguageModelRepository model_repo;

	public LanguageModelService(TokenWeightRepository token_weight_repo,
								GraphLanguageModelRepository model_repo) {
		this.token_weight_repo = token_weight_repo;
		this.model_repo = model_repo;
	}

	/**
	 * Snapshots the current graph into a {@link GraphLanguageModel}. Each source token's
	 *  outgoing edges become a normalized probability distribution; negative weights are
	 *  clamped to zero so a single accidentally negative weight cannot invert the
	 *  distribution. Source tokens whose outgoing weights sum to zero are skipped.
	 */
	public GraphLanguageModel generateModel() {
		List<TokenWeight> edges = token_weight_repo.findAllWithEndpoints();
		log.info("generating language model from {} graph edges", edges == null ? 0 : edges.size());

		Map<String, Map<String, Double>> raw = new LinkedHashMap<>();
		if (edges != null) {
			for (TokenWeight edge : edges) {
				Token source = edge.getToken();
				Token target = edge.getEndToken();
				if (source == null || target == null) {
					continue;
				}
				String src_value = source.getValue();
				String tgt_value = target.getValue();
				if (src_value == null || tgt_value == null) {
					continue;
				}
				double weight = Math.max(edge.getWeight(), 0.0);
				Map<String, Double> row = raw.get(src_value);
				if (row == null) {
					row = new LinkedHashMap<>();
					raw.put(src_value, row);
				}
				Double current = row.get(tgt_value);
				row.put(tgt_value, (current == null ? 0.0 : current) + weight);
			}
		}

		Map<String, Map<String, Double>> normalized = new LinkedHashMap<>();
		int transition_count = 0;
		for (Map.Entry<String, Map<String, Double>> entry : raw.entrySet()) {
			Map<String, Double> row = entry.getValue();
			double sum = 0.0;
			for (Double w : row.values()) {
				sum += w;
			}
			if (sum <= 0.0) {
				continue;
			}
			Map<String, Double> normalized_row = new LinkedHashMap<>();
			for (Map.Entry<String, Double> t : row.entrySet()) {
				normalized_row.put(t.getKey(), t.getValue() / sum);
			}
			normalized.put(entry.getKey(), normalized_row);
			transition_count += normalized_row.size();
		}

		GraphLanguageModel model = new GraphLanguageModel();
		model.setTransitions(normalized);
		model.setVocabularySize(normalized.size());
		model.setTransitionCount(transition_count);
		return model_repo.save(model);
	}

	/**
	 * Returns the most probable next token after {@code seed}, or {@code null} if the seed
	 *  has no outgoing transitions in this model. Ties are broken by insertion order.
	 */
	public String predictNext(GraphLanguageModel model, String seed) {
		if (model == null) {
			throw new IllegalArgumentException("model cannot be null");
		}
		if (seed == null) {
			throw new IllegalArgumentException("seed cannot be null");
		}
		Map<String, Double> row = model.getTransitions().get(seed);
		if (row == null || row.isEmpty()) {
			return null;
		}
		String best = null;
		double best_prob = Double.NEGATIVE_INFINITY;
		for (Map.Entry<String, Double> entry : row.entrySet()) {
			if (entry.getValue() > best_prob) {
				best_prob = entry.getValue();
				best = entry.getKey();
			}
		}
		return best;
	}

	/**
	 * Samples a next token after {@code seed} from its conditional distribution, using
	 *  {@code random} as the source of randomness. Returns {@code null} if the seed has no
	 *  outgoing transitions.
	 */
	public String sampleNext(GraphLanguageModel model, String seed, Random random) {
		if (model == null) {
			throw new IllegalArgumentException("model cannot be null");
		}
		if (seed == null) {
			throw new IllegalArgumentException("seed cannot be null");
		}
		if (random == null) {
			throw new IllegalArgumentException("random cannot be null");
		}
		Map<String, Double> row = model.getTransitions().get(seed);
		if (row == null || row.isEmpty()) {
			return null;
		}
		double r = random.nextDouble();
		double cumulative = 0.0;
		String last = null;
		for (Map.Entry<String, Double> entry : row.entrySet()) {
			cumulative += entry.getValue();
			last = entry.getKey();
			if (r <= cumulative) {
				return last;
			}
		}
		return last;
	}

	/**
	 * Generates a token sequence of at most {@code max_length} tokens by repeatedly choosing
	 *  the next token after the most recent one. When {@code random} is {@code null} the
	 *  next token is chosen greedily; otherwise it is sampled. Generation stops early if a
	 *  token has no outgoing transitions or repeats the immediately previous token.
	 */
	public List<String> generate(GraphLanguageModel model, String seed, int max_length, Random random) {
		if (model == null) {
			throw new IllegalArgumentException("model cannot be null");
		}
		if (seed == null || seed.isEmpty()) {
			throw new IllegalArgumentException("seed cannot be null or empty");
		}
		if (max_length < 1) {
			throw new IllegalArgumentException("max_length must be at least 1");
		}

		List<String> sequence = new ArrayList<>();
		sequence.add(seed);
		String current = seed;
		for (int i = 1; i < max_length; i++) {
			String next = random == null ? predictNext(model, current) : sampleNext(model, current, random);
			if (next == null || next.equals(current)) {
				break;
			}
			sequence.add(next);
			current = next;
		}
		return sequence;
	}
}
