package com.deepthought;

import static org.testng.Assert.assertEquals;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

import org.testng.annotations.Test;

import com.deepthought.models.Token;
import com.deepthought.brain.TokenVector;

@Test(groups = "Regression")
public class TokenVectorTests {

	@Test
	public void loadMarksExistingTokensAsOneAndMissingAsZero() {
		List<Token> input = Arrays.asList(new Token("form"), new Token("button"), new Token("link"));
		List<Token> output = Arrays.asList(new Token("button"), new Token("image"));

		HashMap<String, Integer> record = TokenVector.load(input, output);

		assertEquals(record.get("form").intValue(), 0);
		assertEquals(record.get("button").intValue(), 1);
		assertEquals(record.get("link").intValue(), 0);
	}

	@Test
	public void loadReturnsEmptyMapForEmptyInput() {
		HashMap<String, Integer> record = TokenVector.load(Arrays.asList(), Arrays.asList(new Token("x")));
		assertEquals(record.size(), 0);
	}
}
