package com.deepthought.api;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

import org.json.JSONException;
import org.json.JSONObject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.server.ResponseStatusException;

import com.deepthought.models.Token;
import com.deepthought.models.MemoryRecord;
import com.deepthought.models.edges.Prediction;
import com.deepthought.models.repository.TokenRepository;
import com.deepthought.models.repository.MemoryRecordRepository;
import com.deepthought.models.repository.PredictionRepository;
import com.deepthought.brain.Brain;
import com.deepthought.db.DataDecomposer;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Schema;


/**
 *	API endpoints for learning and making predictions. This set of endpoints allows interacting with the knowledge graph
 *	 to generate, update and retrieve weight matrices(models) for any given input token set and output set
 */
@RestController
@RequestMapping("/rl")
public class ReinforcementLearningController {
	private final Logger log = LoggerFactory.getLogger(this.getClass());

	@Autowired
	private TokenRepository token_repo;

	@Autowired
	private MemoryRecordRepository memory_repo;

	@Autowired
	private PredictionRepository prediction_repo;

	@Autowired
	private Brain brain;

    /**
     * Generates a prediction based on stringified JSON object, input and output {@link Vocabulary}
     * 	labels and any new output tokens the system should predict for. If input passed is not a JSON Object
     *  this endpoint assumes the object is a unstructured {@link String}
     *
     * @param json_obj stringified JSON object containing data that user would like used for prediction
     * @param new_output_tokens
     *
     * @return
     *
     * @throws IllegalArgumentException
     * @throws IllegalAccessException
     * @throws NullPointerException
     * @throws JSONException
     */
	@Operation(summary = "Make a prediction and return a MemoryRecord", description = "", tags = { "Reinforcement Learning" })
    @RequestMapping(value ="/predict", method = RequestMethod.POST)
    public @ResponseBody MemoryRecord predict(@Schema(description = "JSON representation of data", example = "{'field_1':{'field_2':'hello'}}", required = true) @RequestParam(value="input", required=true) String input,
    										  @Schema(description = "List of output labels to be predicted", example = "label_1,label_2,label_n", required = true) @RequestParam(value="output_tokens", required=true) String[] output_labels)
    												  throws IllegalArgumentException, IllegalAccessException, NullPointerException{
		List<Token> input_tokens;
		try {
    		//Break down object into list of tokens
        	input_tokens = DataDecomposer.decompose(new JSONObject(input));
    	}
    	catch(JSONException e) {
    		input_tokens = DataDecomposer.decompose(input);
    	}

    	List<Token> output_tokens = new ArrayList<Token>();
		for(String value : output_labels){
			Token token_record = token_repo.findByValue(value);
			if(token_record==null){
				value = value.replace("[", "");
	        	value = value.replace("]", "");
				Token new_token = new Token(value);
				output_tokens.add(new_token);
			}
			else {
				output_tokens.add(token_record);
			}
		}

    	log.debug("loading vocabulary");
    	//Vocabulary input_vocab = vocabulary_repo.findByLabel(input_vocab_label);
    	//for each token, check if token is in input_vocab
    	List<String> input_token_keys = new ArrayList<String>();

    	List<Token> scrubbed_input_tokens = new ArrayList<Token>();
    	List<String> seen_tokens = new ArrayList<String>();
		for(Token input_token : input_tokens){
			boolean input_equals_output = false;
	    	for(Token output_token : output_tokens){
    			if(output_token.getValue().equalsIgnoreCase(input_token.getValue())){
    				input_equals_output = true;
    			}
    		}

    		if(!input_equals_output && input_token.getValue() != null && !input_token.getValue().equals("null") && !input_token.getValue().trim().isEmpty()
    				&& !seen_tokens.contains(input_token.getValue())){
    			scrubbed_input_tokens.add(input_token);
    		}

    		seen_tokens.add(input_token.getValue());
    	}

    	for(Token token : scrubbed_input_tokens){
    		input_token_keys.add(token.getValue());
    	}

    	//List<Token> output_token = new_tokens;
    	String[] output_token_keys = new String[output_tokens.size()];

    	//load token vector for output_vocab
    	log.debug("loading output token set");

    	//generate policy for input vocab token vector and output vocab token vector
    	double[][] policy = brain.generatePolicy(scrubbed_input_tokens, output_tokens);

    	//generate prediction
    	log.debug("Predicting...  "+policy);
    	double[] prediction = brain.predict(policy);

    	//create prediction edges
    	for(int i=0; i< output_tokens.size(); i++){
    		output_token_keys[i] = output_tokens.get(i).getValue();
      	}

		int max_idx = getMaxPredictionIndex(prediction);

    	//create memory and save vocabularies, policy matrix and prediction vector
    	MemoryRecord memory = new MemoryRecord();
    	memory.setPolicyMatrix(policy);
    	memory.setInputTokenValues(input_token_keys);
    	memory.setOutputTokenKeys(output_token_keys);
    	//memory.setPrediction(prediction);
    	memory.setPredictedToken(output_tokens.get(max_idx));
		memory = memory_repo.save(memory);

		//iterate over tokens to create prediction edges for the memory
		List<Prediction> prediction_edges = new ArrayList<>();
		for(int i=0; i< output_tokens.size(); i++) {
    		Prediction prediction_edge = new Prediction(memory, output_tokens.get(i), prediction[i]);
    		prediction_edges.add(prediction_repo.save(prediction_edge));
		}

		memory.setPredictions(prediction_edges);
       	return memory;
	}

    /**
     *
     * @param memory_id
     * @param token_value
     * @throws JSONException
     * @throws IllegalArgumentException
     * @throws IllegalAccessException
     * @throws NullPointerException
     * @throws IOException
     */
	@Operation(summary = "Applies learning to provided token for a given memory", description = "", tags = { "Reinforcement Learning" })
    @RequestMapping(value ="/learn", method = RequestMethod.POST)
    @ResponseStatus(value = HttpStatus.ACCEPTED, reason = "Successfully learned from feedback")
    public  @ResponseBody void learn(@Schema(description = "unique identifier for specific memory", example = "12345", required = true) @RequestParam(value="memory_id", required=true) long memory_id,
    							 @Schema(description = "value of token that you want to label memory with and learn from", example = "VERB", required = true) @RequestParam(value="token_value", required=true) String token_value)
					 throws JSONException, IllegalArgumentException, IllegalAccessException, NullPointerException, IOException
    {
	    Optional<MemoryRecord> optional_memory = memory_repo.findById(memory_id);
	    if(!optional_memory.isPresent()) {
	    	throw new ResponseStatusException(HttpStatus.NOT_FOUND, "Memory record not found for id " + memory_id);
	    }

	    //log.info("object definition list size :: "+token_list.size());
    	Token token = new Token(token_value);
    	Token token_record = token_repo.findByValue(token.getValue());
    	if(token_record != null){
    		token = token_record;
    	}
    	//LOAD OBJECT DEFINITION LIST BY DECOMPOSING json_string
	    brain.learn(memory_id, token); //token_list, predicted, token, isRewarded);
    }

	static int getMaxPredictionIndex(double[] prediction) {
		if (prediction == null || prediction.length == 0) {
			throw new IllegalArgumentException("Prediction array cannot be null or empty");
		}

		double max_pred = Double.NEGATIVE_INFINITY;
		int max_idx = 0;
		for (int idx = 0; idx < prediction.length; idx++) {
			if (prediction[idx] > max_pred) {
				max_idx = idx;
				max_pred = prediction[idx];
			}
		}
		return max_idx;
	}

	/**
	 *
	 * @param json_object
	 * @param label
	 * @throws JSONException
	 * @throws IllegalArgumentException
	 * @throws IllegalAccessException
	 * @throws NullPointerException
	 * @throws IOException
	 */
	@Operation(summary = "Performs training iteration using label and given object data", description = "", tags = { "Reinforcement Learning" })
    @RequestMapping(value ="/train", method = RequestMethod.POST)
    public  @ResponseBody void train(@RequestParam(value="json_object", required=true) String json_object,
    								 @RequestParam String label)
						 throws JSONException, IllegalArgumentException, IllegalAccessException, NullPointerException, IOException
    {

    	JSONObject json_obj = new JSONObject(json_object);
    	List<Token> token_list = DataDecomposer.decompose(json_obj);
    	brain.train(token_list, label);
    }
}

@ResponseStatus(HttpStatus.INTERNAL_SERVER_ERROR)
class EmptyVocabularyException extends RuntimeException {
	/**
	 *
	 */
	private static final long serialVersionUID = 7200878662560716216L;

	public EmptyVocabularyException() {
		super("Tokens list for output vocabulary cannot be empty or null");
	}
}
