package com.deepthought.models.repository;

import java.util.List;

import org.springframework.data.neo4j.annotation.Query;
import org.springframework.data.neo4j.repository.Neo4jRepository;

import com.deepthought.models.edges.TokenWeight;

/**
 * Spring Data Repository pattern to perform CRUD operations and other various queries
 *  on {@link TokenWeight}
 */
public interface TokenWeightRepository extends Neo4jRepository<TokenWeight, Long> {

	/**
	 * Retrieves every {@link TokenWeight} edge in the graph with both endpoint
	 *  {@link com.deepthought.models.Token Token} nodes populated. Used to build a
	 *  bigram language model from the current learned weights.
	 *
	 * @return list of every persisted HAS_RELATED_TOKEN edge
	 */
	@Query("MATCH (s:Token)-[r:HAS_RELATED_TOKEN]->(t:Token) RETURN s, r, t")
	public List<TokenWeight> findAllWithEndpoints();
}
