package com.deepthought.db;

import java.util.ArrayList;
import java.util.HashMap;


/**
 * A list of unique objects of a designated type that are stored and loaded in a specific order as
 *  a way of maintaining the ability to grow a vertex of tokens that are always in the exact same
 *  order from run to run.
 *
 */
public class VocabularyWeights{

	//private ArrayList<String> valueList = null;
	//private ArrayList<Float> weights = null;
	private HashMap<String, HashMap<String, Float>> vocabulary_weights= null;
	//private ArrayList<ArrayList<Float>> actions = null; 

	private String label = null;
	
	/**
	 * Generates an empty list with the given label as the list name.
	 * 
	 * @param valueList
	 */
	public VocabularyWeights(String listLabel) {
		//this.valueList = new ArrayList<String>();
		this.label = listLabel;
		//this.weights = new ArrayList<Float>();
		this.setVocabulary_weights(new HashMap<String, HashMap<String, Float>>());
	}
	
	/**
	 * A specifically ordered list of values of a certain type specified as the label
	 * 
	 * @param valueList
	 */
	public VocabularyWeights(ArrayList<String> valueList, String label) {
		//this.valueList = valueList;
		this.label = label;
		//this.weights = new ArrayList<Float>(valueList.size());
	}
	
	/**
	 * Appends an object of the vocabulary type to the end of the current vocabulary list if it 
	 *  doesn't yet exist in the valueList
	 * 
	 * @param obj
	 * @return
	 */
	public void appendToVocabulary(String obj){
		if(this.vocabulary_weights.containsKey(obj)){
			return;
		}
		vocabulary_weights.put(obj, new HashMap<String, Float>());
	}
	
	public String getLabel(){
		return this.label;
	}

	/**
	 * Appends a weight to the end of the weights list
	 * @param d
	 */
	public void appendToWeights(String key, String action_key, float d) {
		HashMap<String, Float> action_weights = this.vocabulary_weights.get(key);
		if(!action_weights.containsKey(action_key)){
			action_weights.put(action_key, d);
		}
	}

	public HashMap<String, HashMap<String, Float>> getVocabulary_weights() {
		return vocabulary_weights;
	}

	public void setVocabulary_weights(HashMap<String, HashMap<String, Float>> vocabulary_weights) {
		this.vocabulary_weights = vocabulary_weights;
	}
}
