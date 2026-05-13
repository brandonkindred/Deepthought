package Qanairy.deepthought;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertNull;
import static org.testng.Assert.assertTrue;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

import com.deepthought.models.Token;
import com.deepthought.models.edges.TokenWeight;
import com.deepthought.models.repository.TokenRepository;
import com.qanairy.brain.LanguageModelService;

@Test(groups = "Regression")
public class LanguageModelServiceTests {

	private TokenRepository token_repo;
	private LanguageModelService service;

	@BeforeMethod
	public void setUp() {
		token_repo = mock(TokenRepository.class);
		service = new LanguageModelService(token_repo);
	}

	private TokenWeight edge(String source, String target, double weight) {
		TokenWeight tw = new TokenWeight();
		tw.setToken(new Token(source));
		tw.setEndToken(new Token(target));
		tw.setWeight(weight);
		return tw;
	}

	private Set<TokenWeight> edges(TokenWeight... ws) {
		return new LinkedHashSet<>(Arrays.asList(ws));
	}

	@Test
	public void normalizesOutgoingWeightsForSeed() {
		when(token_repo.getTokenWeights("a")).thenReturn(edges(
				edge("a", "b", 1.0),
				edge("a", "c", 3.0)));

		Map<String, Double> distribution = service.nextTokenDistribution("a");

		assertEquals(distribution.size(), 2);
		assertEquals(distribution.get("b"), 0.25, 1e-9);
		assertEquals(distribution.get("c"), 0.75, 1e-9);
	}

	@Test
	public void clampsNegativeWeightsToZero() {
		when(token_repo.getTokenWeights("a")).thenReturn(edges(
				edge("a", "b", -5.0),
				edge("a", "c", 1.0)));

		Map<String, Double> distribution = service.nextTokenDistribution("a");

		assertEquals(distribution.get("b"), 0.0, 1e-9);
		assertEquals(distribution.get("c"), 1.0, 1e-9);
	}

	@Test
	public void mergesDuplicateTransitionsBySumming() {
		when(token_repo.getTokenWeights("a")).thenReturn(edges(
				edge("a", "b", 1.0),
				edge("a", "b", 3.0)));

		Map<String, Double> distribution = service.nextTokenDistribution("a");

		assertEquals(distribution.size(), 1);
		assertEquals(distribution.get("b"), 1.0, 1e-9);
	}

	@Test
	public void returnsEmptyDistributionWhenSeedHasNoEdges() {
		when(token_repo.getTokenWeights("a")).thenReturn(new HashSet<TokenWeight>());

		Map<String, Double> distribution = service.nextTokenDistribution("a");

		assertTrue(distribution.isEmpty());
	}

	@Test
	public void returnsEmptyDistributionWhenAllWeightsNonPositive() {
		when(token_repo.getTokenWeights("a")).thenReturn(edges(
				edge("a", "b", 0.0),
				edge("a", "c", -1.0)));

		Map<String, Double> distribution = service.nextTokenDistribution("a");

		assertTrue(distribution.isEmpty());
	}

	@Test(expectedExceptions = IllegalArgumentException.class)
	public void nextTokenDistributionRejectsNullSeed() {
		service.nextTokenDistribution(null);
	}

	@Test
	public void predictNextReturnsArgmax() {
		when(token_repo.getTokenWeights("a")).thenReturn(edges(
				edge("a", "b", 1.0),
				edge("a", "c", 5.0)));

		assertEquals(service.predictNext("a"), "c");
	}

	@Test
	public void predictNextReturnsNullForUnknownSeed() {
		when(token_repo.getTokenWeights("z")).thenReturn(new HashSet<TokenWeight>());

		assertNull(service.predictNext("z"));
	}

	@Test
	public void sampleNextIsDeterministicForSeededRandom() {
		when(token_repo.getTokenWeights("a")).thenReturn(edges(
				edge("a", "b", 1.0),
				edge("a", "c", 1.0)));

		String first = service.sampleNext("a", new Random(123L));
		String second = service.sampleNext("a", new Random(123L));
		assertEquals(first, second);
	}

	@Test
	public void generateProducesGreedyChainFromLiveGraph() {
		when(token_repo.getTokenWeights("a")).thenReturn(Collections.singleton(edge("a", "b", 1.0)));
		when(token_repo.getTokenWeights("b")).thenReturn(Collections.singleton(edge("b", "c", 1.0)));
		when(token_repo.getTokenWeights("c")).thenReturn(Collections.singleton(edge("c", "d", 1.0)));
		when(token_repo.getTokenWeights("d")).thenReturn(new HashSet<TokenWeight>());

		List<String> sequence = service.generate("a", 10, null);
		assertEquals(sequence, Arrays.asList("a", "b", "c", "d"));
	}

	@Test
	public void generateStopsAtMaxLength() {
		when(token_repo.getTokenWeights("a")).thenReturn(Collections.singleton(edge("a", "b", 1.0)));
		when(token_repo.getTokenWeights("b")).thenReturn(Collections.singleton(edge("b", "c", 1.0)));

		List<String> sequence = service.generate("a", 2, null);
		assertEquals(sequence, Arrays.asList("a", "b"));
	}

	@Test
	public void generateStopsWhenNextRepeatsCurrent() {
		when(token_repo.getTokenWeights("a")).thenReturn(Collections.singleton(edge("a", "b", 1.0)));
		when(token_repo.getTokenWeights("b")).thenReturn(Collections.singleton(edge("b", "b", 1.0)));

		List<String> sequence = service.generate("a", 10, null);
		assertEquals(sequence, Arrays.asList("a", "b"));
	}

	@Test(expectedExceptions = IllegalArgumentException.class)
	public void generateRejectsNullSeed() {
		service.generate(null, 5, null);
	}

	@Test(expectedExceptions = IllegalArgumentException.class)
	public void generateRejectsNonPositiveMaxLength() {
		service.generate("a", 0, null);
	}
}
