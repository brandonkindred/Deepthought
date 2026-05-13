package com.deepthought.brain;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import com.deepthought.models.Token;
import com.deepthought.models.edges.TokenWeight;
import com.deepthought.models.repository.TokenRepository;

/**
 * Treats the current Neo4j graph as a bigram language model: for any source token, its
 *  outgoing {@code HAS_RELATED_TOKEN} edge weights define an unnormalized distribution over
 *  next tokens. Nothing is persisted — every call reads the live edge weights, so updates
 *  from {@link Brain#learn} are reflected immediately.
 */
@Component
public class LanguageModelService {
	private static final Logger log = LoggerFactory.getLogger(LanguageModelService.class);

	/**
	 * Hard cap on the length of a single {@code /llm/generate} response. Caller-supplied
	 *  {@code max_length} values above this are rejected so a runaway request cannot tie up
	 *  the request thread or Neo4j with an unbounded walk.
	 */
	public static final int MAX_GENERATION_LENGTH = 1000;

	private final TokenRepository token_repo;

	public LanguageModelService(TokenRepository token_repo) {
		this.token_repo = token_repo;
	}

	/**
	 * Returns the normalized probability distribution over next tokens for {@code seed} by
	 *  reading its outgoing edge weights from the graph. Negative weights are clamped to
	 *  zero. The returned map iterates in lexicographic order of target token so seeded
	 *  sampling is reproducible across runs even though the repository returns a {@code Set}
	 *  and the underlying Cypher has no {@code ORDER BY}. Returns an empty map when the
	 *  token has no outgoing edges or every weight is non-positive.
	 */
	public Map<String, Double> nextTokenDistribution(String seed) {
		if (seed == null) {
			throw new IllegalArgumentException("seed cannot be null");
		}
		Set<TokenWeight> weights = token_repo.getTokenWeights(seed);
		if (weights == null || weights.isEmpty()) {
			return new TreeMap<>();
		}

		Map<String, Double> totals = new TreeMap<>();
		double sum = 0.0;
		for (TokenWeight weight : weights) {
			Token end = weight.getEndToken();
			if (end == null || end.getValue() == null) {
				continue;
			}
			double w = Math.max(weight.getWeight(), 0.0);
			Double existing = totals.get(end.getValue());
			totals.put(end.getValue(), (existing == null ? 0.0 : existing) + w);
			sum += w;
		}
		if (sum <= 0.0) {
			return new TreeMap<>();
		}

		Map<String, Double> distribution = new TreeMap<>();
		for (Map.Entry<String, Double> entry : totals.entrySet()) {
			distribution.put(entry.getKey(), entry.getValue() / sum);
		}
		return distribution;
	}

	/**
	 * Returns the highest-probability next token after {@code seed}, or {@code null} if it
	 *  has no outgoing transitions. Ties are broken by lexicographic order of target token.
	 */
	public String predictNext(String seed) {
		Map<String, Double> distribution = nextTokenDistribution(seed);
		if (distribution.isEmpty()) {
			return null;
		}
		String best = null;
		double best_prob = Double.NEGATIVE_INFINITY;
		for (Map.Entry<String, Double> entry : distribution.entrySet()) {
			if (entry.getValue() > best_prob) {
				best_prob = entry.getValue();
				best = entry.getKey();
			}
		}
		return best;
	}

	/**
	 * Samples a next token from the conditional distribution of {@code seed} using
	 *  {@code random}. Returns {@code null} when the seed has no outgoing transitions.
	 */
	public String sampleNext(String seed, Random random) {
		if (random == null) {
			throw new IllegalArgumentException("random cannot be null");
		}
		Map<String, Double> distribution = nextTokenDistribution(seed);
		if (distribution.isEmpty()) {
			return null;
		}
		double r = random.nextDouble();
		double cumulative = 0.0;
		String last = null;
		for (Map.Entry<String, Double> entry : distribution.entrySet()) {
			cumulative += entry.getValue();
			last = entry.getKey();
			if (r <= cumulative) {
				return last;
			}
		}
		return last;
	}

	/**
	 * Generates a sequence of at most {@code max_length} tokens starting from {@code seed}.
	 *  When {@code random} is {@code null} the next token is chosen greedily; otherwise it
	 *  is sampled. Generation stops early when a token has no outgoing transitions, or, in
	 *  greedy mode, when the next token has already appeared in the sequence — greedy
	 *  generation is fully determined by the current token, so any revisit would produce an
	 *  infinite cycle.
	 *
	 * @throws IllegalArgumentException if {@code max_length} exceeds
	 *         {@link #MAX_GENERATION_LENGTH}; that bound prevents a single request from
	 *         tying up the thread and Neo4j when the graph contains long cycles.
	 */
	public List<String> generate(String seed, int max_length, Random random) {
		if (seed == null || seed.isEmpty()) {
			throw new IllegalArgumentException("seed cannot be null or empty");
		}
		if (max_length < 1) {
			throw new IllegalArgumentException("max_length must be at least 1");
		}
		if (max_length > MAX_GENERATION_LENGTH) {
			throw new IllegalArgumentException("max_length must not exceed " + MAX_GENERATION_LENGTH);
		}

		List<String> sequence = new ArrayList<>();
		sequence.add(seed);
		Set<String> visited = new HashSet<>();
		visited.add(seed);
		String current = seed;
		for (int i = 1; i < max_length; i++) {
			String next = random == null ? predictNext(current) : sampleNext(current, random);
			if (next == null) {
				break;
			}
			if (random == null && visited.contains(next)) {
				break;
			}
			sequence.add(next);
			visited.add(next);
			current = next;
		}
		log.debug("generated {} tokens from seed '{}'", sequence.size(), seed);
		return sequence;
	}
}
