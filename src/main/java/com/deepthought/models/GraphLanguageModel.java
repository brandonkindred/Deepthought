package com.deepthought.models;

import java.util.Date;
import java.util.LinkedHashMap;
import java.util.Map;

import org.neo4j.ogm.annotation.GeneratedValue;
import org.neo4j.ogm.annotation.Id;
import org.neo4j.ogm.annotation.NodeEntity;
import org.neo4j.ogm.annotation.Property;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;

import io.swagger.v3.oas.annotations.media.Schema;

/**
 * A bigram-style language model snapshotted from the current {@code HAS_RELATED_TOKEN}
 *  edge weights in the graph. For each source token, outgoing edge weights are normalized
 *  into a probability distribution over next tokens. The full transition table is
 *  Gson-serialized onto a single Neo4j property so the model can be referenced by id
 *  from subsequent generate calls without re-walking the graph.
 */
@NodeEntity
public class GraphLanguageModel {

	@Schema(description = "Unique identifier of the generated language model.", example = "1")
	@Id
	@GeneratedValue
	private Long id;

	@Schema(description = "Timestamp the model was generated.")
	@Property
	private Date created_at;

	@Schema(description = "Number of distinct source tokens with outgoing transitions.", example = "42")
	@Property
	private int vocabulary_size;

	@Schema(description = "Total number of (source, target) transitions stored in the model.", example = "180")
	@Property
	private int transition_count;

	@JsonIgnore
	@Property
	private String transitions_json;

	public GraphLanguageModel() {
		this.created_at = new Date();
		this.transitions_json = "";
	}

	public Long getId() {
		return id;
	}

	public Date getCreatedAt() {
		return created_at;
	}

	public int getVocabularySize() {
		return vocabulary_size;
	}

	public void setVocabularySize(int vocabulary_size) {
		this.vocabulary_size = vocabulary_size;
	}

	public int getTransitionCount() {
		return transition_count;
	}

	public void setTransitionCount(int transition_count) {
		this.transition_count = transition_count;
	}

	/**
	 * Returns the normalized transition table as {@code source -> (target -> probability)}.
	 *  Probabilities for each source row sum to ~1.0. Returns an empty map when the model
	 *  has no persisted transitions.
	 */
	public Map<String, Map<String, Double>> getTransitions() {
		if (transitions_json == null || transitions_json.isEmpty()) {
			return new LinkedHashMap<>();
		}
		Gson gson = new GsonBuilder().create();
		Map<String, Map<String, Double>> parsed = gson.fromJson(transitions_json,
				new TypeToken<LinkedHashMap<String, LinkedHashMap<String, Double>>>(){}.getType());
		return parsed == null ? new LinkedHashMap<String, Map<String, Double>>() : parsed;
	}

	public void setTransitions(Map<String, Map<String, Double>> transitions) {
		if (transitions == null) {
			this.transitions_json = "";
			return;
		}
		Gson gson = new GsonBuilder().create();
		this.transitions_json = gson.toJson(transitions);
	}
}
