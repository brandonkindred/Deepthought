package com.deepthought.models;

import java.util.Date;
import java.util.List;

import org.neo4j.ogm.annotation.GeneratedValue;
import org.neo4j.ogm.annotation.Id;
import org.neo4j.ogm.annotation.NodeEntity;
import org.neo4j.ogm.annotation.Property;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;

/**
 * Persisted binary logistic regression model. Coefficients are stored as a Gson-serialized
 *  String on a single Neo4j property to mirror how {@link MemoryRecord} persists its policy
 *  matrix. The decoded {@code weights} and {@code vocabulary} are exposed for API responses
 *  while the raw JSON properties are hidden from JSON serialization.
 */
@NodeEntity
public class LogisticRegressionModel {

	@Id
	@GeneratedValue
	private Long id;

	@Property
	private Date created_at;

	@JsonIgnore
	@Property
	private String weights_json;

	@Property
	private double intercept;

	@Property
	private int num_features;

	@JsonIgnore
	@Property
	private String vocabulary_json;

	public LogisticRegressionModel() {
		this.created_at = new Date();
		this.weights_json = "";
		this.vocabulary_json = "";
	}

	public Long getId() {
		return id;
	}

	public Date getCreatedAt() {
		return created_at;
	}

	public double getIntercept() {
		return intercept;
	}

	public void setIntercept(double intercept) {
		this.intercept = intercept;
	}

	public int getNumFeatures() {
		return num_features;
	}

	public void setNumFeatures(int num_features) {
		this.num_features = num_features;
	}

	public double[] getWeights() {
		if (weights_json == null || weights_json.isEmpty()) {
			return new double[0];
		}
		Gson gson = new GsonBuilder().create();
		return gson.fromJson(weights_json, double[].class);
	}

	public void setWeights(double[] weights) {
		Gson gson = new GsonBuilder().create();
		this.weights_json = gson.toJson(weights);
	}

	public List<String> getVocabulary() {
		if (vocabulary_json == null || vocabulary_json.isEmpty()) {
			return null;
		}
		Gson gson = new GsonBuilder().create();
		return gson.fromJson(vocabulary_json, new TypeToken<List<String>>(){}.getType());
	}

	public void setVocabulary(List<String> vocabulary) {
		if (vocabulary == null) {
			this.vocabulary_json = "";
			return;
		}
		Gson gson = new GsonBuilder().create();
		this.vocabulary_json = gson.toJson(vocabulary);
	}
}
