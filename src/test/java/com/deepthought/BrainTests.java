package com.deepthought;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertTrue;

import java.util.ArrayList;

import org.testng.annotations.Test;

import com.deepthought.brain.Brain;

@Test(groups = "Regression")
public class BrainTests {

	@Test
	public void predictNormalizesColumnTotals() {
		Brain brain = new Brain();
		double[][] policy = new double[][] { { 2.0, 2.0 }, { 2.0, 6.0 } };

		double[] prediction = brain.predict(policy);

		assertEquals(prediction.length, 2);
		assertEquals(prediction[0], 0.3333333333333333, 0.0000001);
		assertEquals(prediction[1], 0.6666666666666666, 0.0000001);
	}

	@Test
	public void loadVocabulariesReturnsEmptyListWhenNoLabelsProvided() {
		Brain brain = new Brain();
		ArrayList<?> output = brain.loadVocabularies(new String[] {});
		assertTrue(output.isEmpty());
	}
}
