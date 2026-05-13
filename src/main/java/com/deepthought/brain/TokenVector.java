package com.deepthought.brain;

import java.util.HashMap;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import com.deepthought.models.Token;
import com.deepthought.models.Vocabulary;
import com.deepthought.models.repository.TokenRepository;

@Component
public class TokenVector {
	private static Logger log = LoggerFactory.getLogger(TokenVector.class);

	@Autowired
	private static TokenRepository obj_def_repo;

	public static double[][] loadPolicy(List<Token> input_tokens, List<Token> output_tokens, Vocabulary vocab){
		double[][] vocab_policy = new double[input_tokens.size()][output_tokens.size()];


		log.info("concatenating action tokens into 2d array for vocabulary");
		//set output_tokens for object definition to action probabilities
		for(Token def : input_tokens){
			//load action policy for object definition
			obj_def_repo.save(def);
			/*
			Iterator<PolicyEdge> object_action_policy_iter = def.getPolicyEdges().iterator();

			while(object_action_policy_iter.hasNext()){
				PolicyEdge policy_edge = object_action_policy_iter.next();
				Action current_action = action_repo.load(policy_edge.getActionOut());
				int action_idx = vocab.getValueList().indexOf(current_action.getKey());
				if(action_idx >= 0){
					vocab_policy[k][action_idx] = policy_edge.getProbability();
				}
			}
			*/
		}
		return vocab_policy;
	}


	public static HashMap<String, Integer> load(List<Token> input_tokens, List<Token> output_tokens){
		HashMap<String, Integer> vocabulary_record = new HashMap<String, Integer>();

    	for(Token definition : input_tokens){
    		boolean has_match = false;
    		for(Token record_definition : output_tokens){
    			if(record_definition.equals(definition)){
    				vocabulary_record.put(record_definition.getValue(), 1);
    				has_match = true;
    				break;
    			}
    		}

    		if(!has_match){
    			vocabulary_record.put(definition.getValue(), 0);
    		}
       	}

    	return vocabulary_record;
	}
}
