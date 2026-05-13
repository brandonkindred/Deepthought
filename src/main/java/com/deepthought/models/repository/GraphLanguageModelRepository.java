package com.deepthought.models.repository;

import org.springframework.data.neo4j.repository.Neo4jRepository;

import com.deepthought.models.GraphLanguageModel;

/**
 * Spring Data Repository pattern to perform CRUD operations on
 *  {@link GraphLanguageModel} snapshots.
 */
public interface GraphLanguageModelRepository extends Neo4jRepository<GraphLanguageModel, Long> {
}
