package Qanairy.deepthought;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertNotNull;
import static org.testng.Assert.assertNull;
import static org.testng.Assert.assertTrue;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

import com.deepthought.models.GraphLanguageModel;
import com.deepthought.models.Token;
import com.deepthought.models.edges.TokenWeight;
import com.deepthought.models.repository.GraphLanguageModelRepository;
import com.deepthought.models.repository.TokenWeightRepository;
import com.qanairy.brain.LanguageModelService;

@Test(groups = "Regression")
public class LanguageModelServiceTests {

	private TokenWeightRepository token_weight_repo;
	private GraphLanguageModelRepository model_repo;
	private LanguageModelService service;

	@BeforeMethod
	public void setUp() {
		token_weight_repo = mock(TokenWeightRepository.class);
		model_repo = mock(GraphLanguageModelRepository.class);
		when(model_repo.save(any(GraphLanguageModel.class)))
				.thenAnswer(invocation -> invocation.getArgument(0));
		service = new LanguageModelService(token_weight_repo, model_repo);
	}

	private TokenWeight edge(String source, String target, double weight) {
		TokenWeight tw = new TokenWeight();
		tw.setToken(new Token(source));
		tw.setEndToken(new Token(target));
		tw.setWeight(weight);
		return tw;
	}

	@Test
	public void normalizesOutgoingWeightsPerSource() {
		when(token_weight_repo.findAllWithEndpoints()).thenReturn(Arrays.asList(
				edge("a", "b", 1.0),
				edge("a", "c", 3.0),
				edge("d", "e", 2.0)));

		GraphLanguageModel model = service.generateModel();

		Map<String, Map<String, Double>> transitions = model.getTransitions();
		assertEquals(model.getVocabularySize(), 2);
		assertEquals(model.getTransitionCount(), 3);
		assertEquals(transitions.get("a").get("b"), 0.25, 1e-9);
		assertEquals(transitions.get("a").get("c"), 0.75, 1e-9);
		assertEquals(transitions.get("d").get("e"), 1.0, 1e-9);
	}

	@Test
	public void mergesDuplicateTransitionsBySumming() {
		when(token_weight_repo.findAllWithEndpoints()).thenReturn(Arrays.asList(
				edge("a", "b", 1.0),
				edge("a", "b", 3.0)));

		GraphLanguageModel model = service.generateModel();

		Map<String, Double> row = model.getTransitions().get("a");
		assertEquals(row.size(), 1);
		assertEquals(row.get("b"), 1.0, 1e-9);
		assertEquals(model.getTransitionCount(), 1);
	}

	@Test
	public void clampsNegativeWeightsToZero() {
		when(token_weight_repo.findAllWithEndpoints()).thenReturn(Arrays.asList(
				edge("a", "b", -5.0),
				edge("a", "c", 1.0)));

		GraphLanguageModel model = service.generateModel();

		Map<String, Double> row = model.getTransitions().get("a");
		assertEquals(row.get("b"), 0.0, 1e-9);
		assertEquals(row.get("c"), 1.0, 1e-9);
	}

	@Test
	public void skipsSourcesWithAllZeroWeights() {
		when(token_weight_repo.findAllWithEndpoints()).thenReturn(Arrays.asList(
				edge("a", "b", 0.0),
				edge("a", "c", -1.0),
				edge("d", "e", 1.0)));

		GraphLanguageModel model = service.generateModel();

		assertFalse(model.getTransitions().containsKey("a"));
		assertTrue(model.getTransitions().containsKey("d"));
		assertEquals(model.getVocabularySize(), 1);
	}

	@Test
	public void handlesEmptyGraphGracefully() {
		when(token_weight_repo.findAllWithEndpoints()).thenReturn(new ArrayList<TokenWeight>());

		GraphLanguageModel model = service.generateModel();

		assertNotNull(model);
		assertEquals(model.getVocabularySize(), 0);
		assertEquals(model.getTransitionCount(), 0);
		assertTrue(model.getTransitions().isEmpty());
	}

	@Test
	public void predictNextReturnsArgmaxOfRow() {
		when(token_weight_repo.findAllWithEndpoints()).thenReturn(Arrays.asList(
				edge("a", "b", 1.0),
				edge("a", "c", 5.0)));

		GraphLanguageModel model = service.generateModel();

		assertEquals(service.predictNext(model, "a"), "c");
	}

	@Test
	public void predictNextReturnsNullForUnknownSeed() {
		when(token_weight_repo.findAllWithEndpoints()).thenReturn(Collections.singletonList(
				edge("a", "b", 1.0)));

		GraphLanguageModel model = service.generateModel();

		assertNull(service.predictNext(model, "z"));
	}

	@Test
	public void sampleNextIsDeterministicForSeededRandom() {
		when(token_weight_repo.findAllWithEndpoints()).thenReturn(Arrays.asList(
				edge("a", "b", 1.0),
				edge("a", "c", 1.0)));

		GraphLanguageModel model = service.generateModel();

		String first = service.sampleNext(model, "a", new Random(123L));
		String second = service.sampleNext(model, "a", new Random(123L));
		assertEquals(first, second);
	}

	@Test
	public void generateProducesGreedyChain() {
		when(token_weight_repo.findAllWithEndpoints()).thenReturn(Arrays.asList(
				edge("a", "b", 1.0),
				edge("b", "c", 1.0),
				edge("c", "d", 1.0)));

		GraphLanguageModel model = service.generateModel();

		List<String> sequence = service.generate(model, "a", 10, null);
		assertEquals(sequence, Arrays.asList("a", "b", "c", "d"));
	}

	@Test
	public void generateStopsAtMaxLength() {
		when(token_weight_repo.findAllWithEndpoints()).thenReturn(Arrays.asList(
				edge("a", "b", 1.0),
				edge("b", "c", 1.0),
				edge("c", "d", 1.0)));

		GraphLanguageModel model = service.generateModel();

		List<String> sequence = service.generate(model, "a", 2, null);
		assertEquals(sequence, Arrays.asList("a", "b"));
	}

	@Test(expectedExceptions = IllegalArgumentException.class)
	public void generateRejectsNullSeed() {
		GraphLanguageModel model = new GraphLanguageModel();
		service.generate(model, null, 5, null);
	}

	@Test(expectedExceptions = IllegalArgumentException.class)
	public void generateRejectsNonPositiveMaxLength() {
		GraphLanguageModel model = new GraphLanguageModel();
		service.generate(model, "a", 0, null);
	}
}
