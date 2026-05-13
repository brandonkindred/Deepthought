package com.deepthought.models.repository;

import org.springframework.data.neo4j.repository.Neo4jRepository;

import com.deepthought.models.edges.TokenWeight;

/**
 * Spring Data Repository pattern to perform CRUD operations and other various queries
 *  on {@link TokenWeight}
 */
public interface TokenWeightRepository extends Neo4jRepository<TokenWeight, Long> {

}
