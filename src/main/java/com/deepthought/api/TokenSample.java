package com.deepthought.api;

/**
 * DTO for a single labeled training example used by the token-based logistic regression
 *  endpoints. Deserialized from JSON via Gson.
 */
public class TokenSample {
	private String input;
	private int label;

	public TokenSample() {}

	public TokenSample(String input, int label) {
		this.input = input;
		this.label = label;
	}

	public String getInput() {
		return input;
	}

	public void setInput(String input) {
		this.input = input;
	}

	public int getLabel() {
		return label;
	}

	public void setLabel(int label) {
		this.label = label;
	}
}
