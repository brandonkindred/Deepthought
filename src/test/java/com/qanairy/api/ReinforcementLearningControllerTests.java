package com.qanairy.api;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertNotNull;
import static org.testng.Assert.assertTrue;
import static org.testng.Assert.fail;

import java.lang.reflect.Field;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;

import org.mockito.ArgumentCaptor;
import org.springframework.web.server.ResponseStatusException;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

import com.deepthought.models.Token;
import com.deepthought.models.MemoryRecord;
import com.deepthought.models.edges.Prediction;
import com.deepthought.models.repository.TokenRepository;
import com.deepthought.models.repository.MemoryRecordRepository;
import com.deepthought.models.repository.PredictionRepository;
import com.qanairy.brain.Brain;

@Test(groups = "Regression")
public class ReinforcementLearningControllerTests {

	private ReinforcementLearningController controller;
	private TokenRepository token_repo;
	private MemoryRecordRepository memory_repo;
	private PredictionRepository prediction_repo;
	private Brain brain;

	@BeforeMethod
	public void setUp() throws Exception {
		controller = new ReinforcementLearningController();
		token_repo = mock(TokenRepository.class);
		memory_repo = mock(MemoryRecordRepository.class);
		prediction_repo = mock(PredictionRepository.class);
		brain = mock(Brain.class);

		setField("token_repo", token_repo);
		setField("memory_repo", memory_repo);
		setField("prediction_repo", prediction_repo);
		setField("brain", brain);

		when(memory_repo.save(any(MemoryRecord.class))).thenAnswer(invocation -> invocation.getArgument(0));
		when(prediction_repo.save(any())).thenAnswer(invocation -> invocation.getArgument(0));
	}

	private void setField(String name, Object value) throws Exception {
		Field f = ReinforcementLearningController.class.getDeclaredField(name);
		f.setAccessible(true);
		f.set(controller, value);
	}

	@Test
	public void getMaxPredictionIndex_handlesAllNegativeValues() {
		double[] prediction = new double[] { -0.5, -0.2, -0.9 };
		int idx = ReinforcementLearningController.getMaxPredictionIndex(prediction);
		assertEquals(idx, 1);
	}

	@Test(expectedExceptions = IllegalArgumentException.class)
	public void getMaxPredictionIndex_rejectsNullArray() {
		ReinforcementLearningController.getMaxPredictionIndex(null);
	}

	@Test(expectedExceptions = IllegalArgumentException.class)
	public void getMaxPredictionIndex_rejectsEmptyArray() {
		ReinforcementLearningController.getMaxPredictionIndex(new double[] {});
	}

	@Test
	public void learn_returns404WhenMemoryDoesNotExist() throws Exception {
		when(memory_repo.findById(999L)).thenReturn(Optional.empty());

		try {
			controller.learn(999L, "missing");
			fail("Expected ResponseStatusException");
		} catch (ResponseStatusException e) {
			assertEquals(e.getStatus().value(), 404);
			assertTrue(e.getReason().contains("999"));
		}
	}

	@Test
	public void learn_usesExistingTokenWhenPresent() throws Exception {
		when(memory_repo.findById(123L)).thenReturn(Optional.of(new MemoryRecord()));
		Token existingToken = new Token("existing");
		when(token_repo.findByValue("existing")).thenReturn(existingToken);

		controller.learn(123L, "existing");

		verify(brain).learn(123L, existingToken);
	}

	@Test
	public void learn_createsTokenWhenMissing() throws Exception {
		when(memory_repo.findById(321L)).thenReturn(Optional.of(new MemoryRecord()));
		when(token_repo.findByValue("new-token")).thenReturn(null);

		controller.learn(321L, "new-token");

		verify(brain).learn(eq(321L), any(Token.class));
	}

	@Test
	public void train_decomposesJsonAndDelegatesToBrain() throws Exception {
		controller.train("{\"sentence\":\"hello world\"}", "GREETING");

		verify(brain).train(any(), eq("GREETING"));
	}

	@Test
	public void predict_acceptsJsonInputAndBuildsMemoryRecord() throws Exception {
		Token existingOutput = new Token("known_output");
		when(token_repo.findByValue("known_output")).thenReturn(existingOutput);
		when(token_repo.findByValue("new_output")).thenReturn(null);
		when(brain.generatePolicy(any(), any())).thenReturn(new double[][] { { 0.1, 0.9 } });
		when(brain.predict(any())).thenReturn(new double[] { 0.2, 0.8 });

		MemoryRecord memory = controller.predict("{\"text\":\"alpha beta\"}", new String[] { "known_output", "new_output" });

		assertNotNull(memory);
		assertNotNull(memory.getPredictedToken());
		assertEquals(memory.getPredictedToken().getValue(), "new_output");
		assertEquals(memory.getPredictions().size(), 2);
		assertEquals(memory.getOutputTokenKeys(), new String[] { "known_output", "new_output" });
	}

	@Test
	public void predict_acceptsPlainTextFallbackWhenInputIsNotJson() throws Exception {
		when(token_repo.findByValue("label")).thenReturn(new Token("label"));
		when(brain.generatePolicy(any(), any())).thenReturn(new double[][] { { 1.0 } });
		when(brain.predict(any())).thenReturn(new double[] { 1.0 });

		MemoryRecord memory = controller.predict("plain text input", new String[] { "label" });

		assertNotNull(memory);
		assertEquals(memory.getPredictions().size(), 1);
	}

	@Test
	public void predict_scrubsDuplicateAndInvalidInputTokens() throws Exception {
		when(token_repo.findByValue("hello")).thenReturn(new Token("hello"));
		when(brain.generatePolicy(any(), any())).thenReturn(new double[][] { { 1.0 } });
		when(brain.predict(any())).thenReturn(new double[] { 1.0 });

		MemoryRecord memory = controller.predict("{\"a\":\"hello\",\"b\":\"hello\",\"c\":\"null\",\"d\":\"\"}", new String[] { "hello" });

		assertNotNull(memory);
		assertTrue(memory.getInputTokenValues().isEmpty());
		verify(brain).generatePolicy(any(), any());
		verify(brain, never()).learn(any(Long.class), any(Token.class));
	}

	@Test
	public void predict_stripsBracketCharactersFromUnknownOutputLabels() throws Exception {
		when(token_repo.findByValue("[fresh]"))
				.thenReturn(null);
		when(brain.generatePolicy(any(), any())).thenReturn(new double[][] { { 1.0 } });
		when(brain.predict(any())).thenReturn(new double[] { 1.0 });

		MemoryRecord memory = controller.predict("{\"a\":\"value\"}", new String[] { "[fresh]" });

		assertNotNull(memory);
		assertEquals(Arrays.asList(memory.getOutputTokenKeys()), Arrays.asList("fresh"));
	}

	@Test
	public void predict_persistsPredictionEdgeForEachOutputToken() throws Exception {
		when(token_repo.findByValue(any())).thenReturn(null);
		when(brain.generatePolicy(any(), any())).thenReturn(new double[][] { { 0.1, 0.5, 0.4 } });
		when(brain.predict(any())).thenReturn(new double[] { 0.1, 0.5, 0.4 });

		MemoryRecord memory = controller.predict("{\"a\":\"word\"}", new String[] { "first", "second", "third" });

		assertNotNull(memory);
		assertEquals(memory.getPredictions().size(), 3);
		verify(prediction_repo, times(3)).save(any(Prediction.class));
		verify(memory_repo, times(1)).save(any(MemoryRecord.class));
	}

	@Test
	public void predict_persistsMemoryRecordWithPolicyMatrixAndKeys() throws Exception {
		when(token_repo.findByValue(any())).thenReturn(null);
		double[][] policy = new double[][] { { 0.25, 0.75 } };
		when(brain.generatePolicy(any(), any())).thenReturn(policy);
		when(brain.predict(any())).thenReturn(new double[] { 0.25, 0.75 });

		controller.predict("{\"a\":\"word\"}", new String[] { "[alpha]", "beta" });

		ArgumentCaptor<MemoryRecord> memoryCaptor = ArgumentCaptor.forClass(MemoryRecord.class);
		verify(memory_repo).save(memoryCaptor.capture());
		MemoryRecord saved = memoryCaptor.getValue();
		assertTrue(Arrays.deepEquals(saved.getPolicyMatrix(), policy));
		assertEquals(saved.getInputTokenValues(), Arrays.asList("word"));
		assertEquals(Arrays.asList(saved.getOutputTokenKeys()), Arrays.asList("alpha", "beta"));
	}

	@Test
	@SuppressWarnings("unchecked")
	public void predict_scrubsInputTokenWhenItEqualsOutputTokenCaseInsensitively() throws Exception {
		when(token_repo.findByValue("hello")).thenReturn(new Token("hello"));
		when(brain.generatePolicy(any(), any())).thenReturn(new double[][] { { 1.0 } });
		when(brain.predict(any())).thenReturn(new double[] { 1.0 });

		controller.predict("{\"a\":\"Hello\"}", new String[] { "hello" });

		ArgumentCaptor<List<Token>> inputCaptor = ArgumentCaptor.forClass(List.class);
		verify(brain).generatePolicy(inputCaptor.capture(), any());
		assertTrue(inputCaptor.getValue().isEmpty());
	}

	@Test
	@SuppressWarnings("unchecked")
	public void predict_passesScrubbedInputsToBrainGeneratePolicy() throws Exception {
		when(token_repo.findByValue(any())).thenReturn(null);
		when(brain.generatePolicy(any(), any())).thenReturn(new double[][] { { 1.0 } });
		when(brain.predict(any())).thenReturn(new double[] { 1.0 });

		controller.predict("{\"a\":\"keep\",\"b\":\"null\",\"c\":\"\"}", new String[] { "out" });

		ArgumentCaptor<List<Token>> inputCaptor = ArgumentCaptor.forClass(List.class);
		verify(brain).generatePolicy(inputCaptor.capture(), any());
		List<Token> scrubbed = inputCaptor.getValue();
		assertEquals(scrubbed.size(), 1);
		assertEquals(scrubbed.get(0).getValue(), "keep");
	}

	@Test
	public void predict_picksFirstIndexOnTiedProbabilities() throws Exception {
		when(token_repo.findByValue(any())).thenReturn(null);
		when(brain.generatePolicy(any(), any())).thenReturn(new double[][] { { 0.5, 0.5 } });
		when(brain.predict(any())).thenReturn(new double[] { 0.5, 0.5 });

		MemoryRecord memory = controller.predict("{\"a\":\"word\"}", new String[] { "first", "second" });

		assertNotNull(memory.getPredictedToken());
		assertEquals(memory.getPredictedToken().getValue(), "first");
	}

	@Test
	public void predict_predictionEdgeWeightsMatchBrainOutputInOrder() throws Exception {
		when(token_repo.findByValue(any())).thenReturn(null);
		when(brain.generatePolicy(any(), any())).thenReturn(new double[][] { { 0.3, 0.7 } });
		when(brain.predict(any())).thenReturn(new double[] { 0.3, 0.7 });

		MemoryRecord memory = controller.predict("{\"a\":\"word\"}", new String[] { "low", "high" });

		List<Prediction> edges = memory.getPredictions();
		assertEquals(edges.size(), 2);
		assertEquals(edges.get(0).getWeight(), 0.3);
		assertEquals(edges.get(0).getToken().getValue(), "low");
		assertEquals(edges.get(1).getWeight(), 0.7);
		assertEquals(edges.get(1).getToken().getValue(), "high");
	}

	@Test(expectedExceptions = IllegalArgumentException.class)
	public void predict_throwsWhenNoOutputLabelsProvided() throws Exception {
		when(brain.generatePolicy(any(), any())).thenReturn(new double[][] {});
		when(brain.predict(any())).thenReturn(new double[] {});

		controller.predict("{\"a\":\"word\"}", new String[] {});
	}

	@Test
	public void predict_decomposesNestedJsonInput() throws Exception {
		when(token_repo.findByValue(any())).thenReturn(null);
		when(brain.generatePolicy(any(), any())).thenReturn(new double[][] { { 1.0 } });
		when(brain.predict(any())).thenReturn(new double[] { 1.0 });

		controller.predict("{\"outer\":{\"inner\":\"deepvalue\"},\"sibling\":\"flatvalue\"}", new String[] { "out" });

		ArgumentCaptor<MemoryRecord> memoryCaptor = ArgumentCaptor.forClass(MemoryRecord.class);
		verify(memory_repo).save(memoryCaptor.capture());
		List<String> inputs = memoryCaptor.getValue().getInputTokenValues();
		assertTrue(inputs.contains("deepvalue"));
		assertTrue(inputs.contains("flatvalue"));
	}

	@Test
	public void predict_preservesMultipleScrubbedInputTokens() throws Exception {
		when(token_repo.findByValue(any())).thenReturn(null);
		when(brain.generatePolicy(any(), any())).thenReturn(new double[][] { { 1.0 } });
		when(brain.predict(any())).thenReturn(new double[] { 1.0 });

		controller.predict("{\"a\":\"alpha\",\"b\":\"beta\",\"c\":\"gamma\"}", new String[] { "out" });

		ArgumentCaptor<MemoryRecord> memoryCaptor = ArgumentCaptor.forClass(MemoryRecord.class);
		verify(memory_repo).save(memoryCaptor.capture());
		List<String> inputs = memoryCaptor.getValue().getInputTokenValues();
		assertEquals(inputs.size(), 3);
		assertTrue(inputs.contains("alpha"));
		assertTrue(inputs.contains("beta"));
		assertTrue(inputs.contains("gamma"));
	}

	@Test
	public void predict_reusesExistingOutputTokenInstance() throws Exception {
		Token existing = new Token("foo");
		when(token_repo.findByValue("foo")).thenReturn(existing);
		when(brain.generatePolicy(any(), any())).thenReturn(new double[][] { { 1.0 } });
		when(brain.predict(any())).thenReturn(new double[] { 1.0 });

		MemoryRecord memory = controller.predict("{\"a\":\"x\"}", new String[] { "foo" });

		assertTrue(memory.getPredictedToken() == existing);
		assertEquals(memory.getPredictions().size(), 1);
		assertTrue(memory.getPredictions().get(0).getToken() == existing);
	}
}
