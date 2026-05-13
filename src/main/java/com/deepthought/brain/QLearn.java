package com.deepthought.brain;

/**
 * Provides q-learning equation using a constant learning_rate and discount_factor
 *
 */
public class QLearn {

	private final double learning_rate;
	private final double discount_factor;
	
	/**
	 * 
	 * @param learning_rate
	 * @param discount_factor
	 */
	public QLearn(double learning_rate, double discount_factor) {
		this.learning_rate = learning_rate;
		this.discount_factor = discount_factor;
	}
	
	/**
	 * Calculates new value using q-learning equation
	 * 
	 * @param old_value value experienced previously
	 * @param actual_reward
	 * @param estimated_future_reward predicted reward
	 * @return
	 */
	public double calculate(double old_value, double actual_reward, double estimated_future_reward){
		return (old_value + learning_rate * (actual_reward + (discount_factor * estimated_future_reward)));
	}

}
