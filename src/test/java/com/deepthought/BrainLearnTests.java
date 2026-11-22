package com.deepthought;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyDouble;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertSame;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Optional;

import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;
import org.springframework.test.util.ReflectionTestUtils;

import com.deepthought.brain.Brain;
import com.deepthought.models.MemoryRecord;
import com.deepthought.models.Token;
import com.deepthought.models.edges.TokenWeight;
import com.deepthought.models.repository.MemoryRecordRepository;
import com.deepthought.models.repository.TokenRepository;
import com.deepthought.models.repository.TokenWeightRepository;

/**
 * Unit tests for {@link Brain#learn(long, Token)} using mocked repositories.
 * Q-update is: new = old + 0.1 * (reward + 0.1 * 1.0), clamped to [0, 1]
 */
@Test(groups = "Regression")
public class BrainLearnTests {
	private static final double DELTA = 0.000000001;
	private static final long MEMORY_ID = 1L;

	private TokenRepository token_repo;
	private TokenWeightRepository token_weight_repo;
	private MemoryRecordRepository memory_repo;
	private Brain brain;
	private MemoryRecord memory;

	@BeforeMethod
	public void setUp() {
		token_repo = mock(TokenRepository.class);
		token_weight_repo = mock(TokenWeightRepository.class);
		memory_repo = mock(MemoryRecordRepository.class);

		brain = new Brain();
		ReflectionTestUtils.setField(brain, "token_repo", token_repo);
		ReflectionTestUtils.setField(brain, "token_weight_repo", token_weight_repo);
		ReflectionTestUtils.setField(brain, "memory_repo", memory_repo);

		memory = new MemoryRecord();
		memory.setInputTokenValues(Arrays.asList("in"));
		memory.setOutputTokenKeys(new String[] { "x", "y" });
		memory.setPredictedToken(new Token("x"));
		when(memory_repo.findById(MEMORY_ID)).thenReturn(Optional.of(memory));
		when(token_repo.save(any(Token.class))).thenAnswer(inv -> inv.getArgument(0));
	}

	/** Registers an existing in -> output_key edge with the given weight and returns it */
	private TokenWeight stubEdge(String output_key, double weight) {
		TokenWeight edge = new TokenWeight();
		edge.setWeight(weight);
		edge.setEndToken(new Token(output_key));
		Token in = new Token("in", new ArrayList<TokenWeight>(Collections.singletonList(edge)));
		when(token_repo.getConnectedTokens("in", output_key)).thenReturn(Collections.singletonList(in));
		return edge;
	}

	@Test
	public void correctPredictionRewardsPredictedEdgeAndSkipsUnrelatedOutputs() throws Exception {
		TokenWeight x_edge = stubEdge("x", 0.5);

		brain.learn(MEMORY_ID, new Token("x"));

		assertEquals(x_edge.getWeight(), 0.71, DELTA); // reward +2
		verify(token_weight_repo).save(x_edge);
		// "y" is neither predicted nor actual, so it must not be touched
		verify(token_repo, never()).getConnectedTokens("in", "y");
		verify(token_repo, never()).createWeightedConnection(eq("in"), eq("y"), anyDouble());
	}

	@Test
	public void wrongPredictionRewardsActualAndPenalizesPredicted() throws Exception {
		TokenWeight x_edge = stubEdge("x", 0.5);
		TokenWeight y_edge = stubEdge("y", 0.5);

		brain.learn(MEMORY_ID, new Token("y"));

		assertEquals(y_edge.getWeight(), 0.61, DELTA); // actual but not predicted: +1
		assertEquals(x_edge.getWeight(), 0.41, DELTA); // predicted but not actual: -1
		verify(token_weight_repo).save(x_edge);
		verify(token_weight_repo).save(y_edge);
	}

	@Test
	public void weightNeverDropsBelowZero() throws Exception {
		TokenWeight x_edge = stubEdge("x", 0.05);
		stubEdge("y", 0.5);

		brain.learn(MEMORY_ID, new Token("y")); // x is predicted but wrong: -1 -> 0.05 - 0.09

		assertEquals(x_edge.getWeight(), 0.0, DELTA);
	}

	@Test
	public void weightNeverExceedsOne() throws Exception {
		TokenWeight x_edge = stubEdge("x", 0.95);

		brain.learn(MEMORY_ID, new Token("x")); // +2 -> 0.95 + 0.21

		assertEquals(x_edge.getWeight(), 1.0, DELTA);
	}

	@Test
	public void createsMissingEdgeThenUpdatesIt() throws Exception {
		when(token_repo.getConnectedTokens("in", "x")).thenReturn(Collections.<Token>emptyList());
		TokenWeight created = new TokenWeight();
		created.setWeight(0.3);
		when(token_repo.createWeightedConnection(eq("in"), eq("x"), anyDouble())).thenReturn(created);

		brain.learn(MEMORY_ID, new Token("x"));

		assertEquals(created.getWeight(), 0.51, DELTA); // 0.3 + 0.21
		verify(token_weight_repo).save(created);
	}

	@Test
	public void updatesTheEdgeMatchingTheOutputNotJustTheFirstEdge() throws Exception {
		TokenWeight other = new TokenWeight();
		other.setWeight(0.9);
		other.setEndToken(new Token("other"));
		TokenWeight x_edge = new TokenWeight();
		x_edge.setWeight(0.5);
		x_edge.setEndToken(new Token("x"));
		Token in = new Token("in", new ArrayList<TokenWeight>(Arrays.asList(other, x_edge)));
		when(token_repo.getConnectedTokens("in", "x")).thenReturn(Collections.singletonList(in));

		brain.learn(MEMORY_ID, new Token("x"));

		assertEquals(x_edge.getWeight(), 0.71, DELTA);
		assertEquals(other.getWeight(), 0.9, DELTA);
		verify(token_weight_repo, never()).save(other);
	}

	@Test
	public void persistsDesiredTokenOnMemoryRecord() throws Exception {
		stubEdge("x", 0.5);
		stubEdge("y", 0.5);
		Token existing = new Token("y");
		when(token_repo.findByValue("y")).thenReturn(existing);

		brain.learn(MEMORY_ID, new Token("y"));

		assertSame(memory.getDesiredToken(), existing);
		verify(memory_repo).save(memory);
	}

	@Test
	public void savesNewDesiredTokenWhenNotInGraph() throws Exception {
		stubEdge("x", 0.5);
		when(token_repo.findByValue(anyString())).thenReturn(null);
		Token actual = new Token("x");

		brain.learn(MEMORY_ID, actual);

		verify(token_repo).save(actual);
		assertSame(memory.getDesiredToken(), actual);
	}
}
