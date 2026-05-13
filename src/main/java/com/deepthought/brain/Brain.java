package com.deepthought.brain;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.Random;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import com.deepthought.models.Token;
import com.deepthought.models.MemoryRecord;
import com.deepthought.models.Vocabulary;
import com.deepthought.models.edges.TokenWeight;
import com.deepthought.models.repository.TokenRepository;
import com.deepthought.models.repository.TokenWeightRepository;
import com.deepthought.models.repository.MemoryRecordRepository;

import edu.stanford.nlp.util.ArrayUtils;

/**
 * Provides ability to predict and learn from data
 *
 */
@Component
public class Brain {
	private static Logger log = LoggerFactory.getLogger(Brain.class);

	@Autowired
	private TokenRepository token_repo;

	@Autowired
	private TokenWeightRepository token_weight_repo;

	@Autowired
	private MemoryRecordRepository memory_repo;

	public Brain(){}

	public double[] predict(double[][] policy){

		double[] prediction = new double[policy[0].length];
		for(int idx1 = 0; idx1 < policy[0].length; idx1++){
			double sum = 0.0;
			for(int idx2 = 0; idx2 < policy.length; idx2++){
				sum += policy[idx2][idx1];
			}

			prediction[idx1] = sum;
		}
		prediction = ArrayUtils.normalize(prediction);

		return prediction;
	}

	/**
	 * Reads path and performs learning tasks
	 *
	 *
		//Learning process
		//	1. identify vocabulary that this set of object info belongs to
		//  2. Create record based on vocabulary
		//  3. load known vocabulary action policies
		//  4. perform matrix math to inline update vocabulary policies for record based on reward/penalty for productivity
	 *
	 * @param path an {@link ArrayList} of graph vertex indices. Order matters
	 * @throws IllegalArgumentException
	 * @throws IllegalAccessException
	 * @throws NullPointerException
	 * @throws IOException
	 */
	public void learn(long memory_id,
					  Token actual_token)
						  throws IllegalArgumentException, IllegalAccessException,
							  NullPointerException, IOException{
		//Load memory from database
		Optional<MemoryRecord> memory_record = memory_repo.findById(memory_id);

		// 2a. load known action policies/probabilities for each object definition in the definition list
		MemoryRecord memory = memory_record.get();

		// 3. determine reward/regret score based on productivity status
		//Q-LEARNING VARIABLES
		final double learning_rate = .1;
		final double discount_factor = .1;

		//set estimated reward using prediction from memory.

		//replace with steps to estimate reward for an output token independent of actual desired output token
		double estimated_reward = 1.0;

		// 3. determine reward/regret score based on productivity status
		double actual_reward = 0.0;
		QLearn q_learn = new QLearn(learning_rate, discount_factor);
		for(String output_key : memory.getOutputTokenKeys()){

			//if predicted token is equal to output token and actual token is equal to predicted token  OR output key equals actual token key
			if(output_key.equals(actual_token.getValue()) && actual_token.getValue().equals(memory.getPredictedToken().getValue())){
				log.debug("REWARD   ::    2");
				actual_reward = 2.0;
			}
			else if(output_key.equals(actual_token.getValue())){
				log.debug("REWARD   ::   1");
				actual_reward = 1.0;
			}
			//if output isn't equal to the actual token or the predicted token, don't affect weights
			else if(output_key.equals(memory.getPredictedToken().getValue()) && !output_key.equals(actual_token.getValue())){
				log.debug("REWARD   ::     -2");
				actual_reward = -1.0;
			}
			else if(!output_key.equals(actual_token.getValue())) {
				log.debug("REWARD   ::     -1");
				actual_reward = -2.0;
			}
			else {
				log.debug("REWARD   ::    0");
				//nothing changed so there was no reward for that combination. We want to remember this in the future
				// so we set it to a negative value to simulate regret
				actual_reward = 0.0;
			}

			List<TokenWeight> token_weights = new ArrayList<TokenWeight>();
			for(String input_key : memory.getInputTokenValues()){
				memory.setDesiredToken(actual_token);
				log.info("input key :: "+input_key);
				log.info("output key :: " + output_key);
				List<Token> tokens = token_repo.getConnectedTokens(input_key, output_key);
				TokenWeight token_weight = null;
				if(tokens.isEmpty()) {
					Random random = new Random();
					double weight = random.nextDouble();

					token_weight = token_repo.createWeightedConnection(input_key, output_key, weight);
				}
				else {
					token_weight = tokens.get(0).getTokenWeights().get(0);
				}
				double q_learn_val = Math.abs(q_learn.calculate(token_weight.getWeight(), actual_reward, estimated_reward ));
				//updated token weight with q_learn_val
				token_weight.setWeight(q_learn_val);
				token_weights.add(token_weight);
				log.debug("token ::    " + token_weight.getToken().getValue() + "  :::   " + token_weight.getWeight());
				token_weight_repo.save(token_weight);
			}
		}
	}

	/**
	 *
	 * @param object_list
	 * @return
	 */
	private Vocabulary predictVocabulary(List<Token> object_list){
		return null;
	}

	/**
	 *
	 * @param object_list
	 * @param vocabulary
	 * @return
	 */
	private List<Token> generateVocabRecord(List<Token> object_list, Vocabulary vocabulary){
		return object_list;
	}

	/**
	 * Retrieves all {@linkplain Vocabulary vocabularies} that are required by the agent
	 *
	 * @param vocabLabels
	 * @return
	 */
	public ArrayList<Vocabulary> loadVocabularies(String[] vocabLabels){
		ArrayList<Vocabulary> vocabularies = new ArrayList<Vocabulary>();
		for(String label : vocabLabels){
			//vocabularies.add(Vocabulary.load(label));
		}
		return vocabularies;
	}

	/**
	 *
	 * @param object_list
	 * @param vocabulary
	 * @return
	 */
	private List<List<Token>> loadActionPolicies(List<Token> object_list, Vocabulary vocabulary){
		return null;

	}

	/**
	 *
	 */
	private void backPropogate(){

	}

	/**
	 *
	 */
	private void saveActionPolicies(){

	}

	public double[][] generatePolicy(List<Token> input_tokens, List<Token> output_tokens){
		// 1. Create a memory record for prediction
		Random random = new Random();
		double[][] policy = new double[input_tokens.size()][output_tokens.size()];
		log.info("###################################################################");
		log.info("input tokens size :: "+input_tokens.size());
		log.info("output tokens size :: "+output_tokens.size());

		for(int in_idx = 0; in_idx < input_tokens.size(); in_idx++){
			for(int out_idx = 0; out_idx < output_tokens.size(); out_idx++){
				List<Token> tokens = token_repo.getConnectedTokens(input_tokens.get(in_idx).getValue(), output_tokens.get(out_idx).getValue());
				double weight = -1.0;

				if(!tokens.isEmpty()){
					for(TokenWeight token_weight : tokens.get(0).getTokenWeights()){
						if(token_weight.getEndToken().equals(output_tokens.get(out_idx))){
							weight = token_weight.getWeight();
						}
					}
				}
				else{
					weight = random.nextDouble();
					TokenWeight token_weight = new TokenWeight();
					token_weight.setEndToken(output_tokens.get(out_idx));
					token_weight.setWeight(weight);
					token_weight.setToken(input_tokens.get(in_idx));
					Token input_token = input_tokens.get(in_idx);
					Token input_token_record = token_repo.findByValue(input_token.getValue());
					if(input_token_record != null){
						input_token = input_token_record;
					}

					input_token.getTokenWeights().add(token_weight);
					token_repo.save(input_token);
				}

				policy[in_idx][out_idx] = weight;
			}
		}
		log.info("###################################################################");


		return policy;
	}

	public void train(List<Token> token_list, String label) {
		//REINFORCEMENT LEARNING
		log.info( " Initiating learning");

		//learning model
		// 1. identify vocabulary (NOTE: This is currently hard coded since we only currently care about 1 context)
		Vocabulary vocabulary = new Vocabulary(new ArrayList<Token>(), "internet");

		log.info("object definition list size :: "+token_list.size());
		// 2. create record based on vocabulary
		for(Token token : token_list){
			vocabulary.appendToVocabulary(token);
		}

		log.info("vocabulary :: "+vocabulary);
		log.info("vocab value list size   :: "+vocabulary.getTokens().size());
		// 2. create state vertex from vocabulary
		int idx = 0;
		for(Token vocab_token : vocabulary.getTokens()){
			boolean[] state = new boolean[vocabulary.getTokens().size()];
			if(token_list.contains(vocab_token)){
				state[idx] = true;
			}
			else{
				state[idx] = false;
			}
			idx++;
		}
	}

}
