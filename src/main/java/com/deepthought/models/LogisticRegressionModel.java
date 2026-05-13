package com.deepthought.models;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Base64;
import java.util.Date;
import java.util.List;

import org.neo4j.ogm.annotation.GeneratedValue;
import org.neo4j.ogm.annotation.Id;
import org.neo4j.ogm.annotation.NodeEntity;
import org.neo4j.ogm.annotation.Property;
import org.tribuo.Model;
import org.tribuo.classification.Label;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;

/**
 * Persisted binary logistic regression model. The fitted Tribuo {@link Model} is
 *  Java-serialized and stored as a Base64 string on a single Neo4j property so we don't
 *  have to round-trip individual coefficients. The vocabulary (when token-trained) is
 *  persisted as a Gson-serialized list to mirror how {@link MemoryRecord} stores its
 *  policy matrix.
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
	private String model_bytes;

	@Property
	private int num_features;

	@JsonIgnore
	@Property
	private String vocabulary_json;

	public LogisticRegressionModel() {
		this.created_at = new Date();
		this.model_bytes = "";
		this.vocabulary_json = "";
	}

	public Long getId() {
		return id;
	}

	public Date getCreatedAt() {
		return created_at;
	}

	public int getNumFeatures() {
		return num_features;
	}

	public void setNumFeatures(int num_features) {
		this.num_features = num_features;
	}

	@JsonIgnore
	@SuppressWarnings("unchecked")
	public Model<Label> getTribuoModel() {
		if (model_bytes == null || model_bytes.isEmpty()) {
			return null;
		}
		byte[] bytes = Base64.getDecoder().decode(model_bytes);
		try (ObjectInputStream in = new ObjectInputStream(new ByteArrayInputStream(bytes))) {
			return (Model<Label>) in.readObject();
		} catch (IOException | ClassNotFoundException e) {
			throw new IllegalStateException("failed to deserialize stored Tribuo model", e);
		}
	}

	public void setTribuoModel(Model<Label> model) {
		if (model == null) {
			this.model_bytes = "";
			return;
		}
		try (ByteArrayOutputStream baos = new ByteArrayOutputStream();
				ObjectOutputStream out = new ObjectOutputStream(baos)) {
			out.writeObject(model);
			out.flush();
			this.model_bytes = Base64.getEncoder().encodeToString(baos.toByteArray());
		} catch (IOException e) {
			throw new IllegalStateException("failed to serialize Tribuo model", e);
		}
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
