package com.deepthought.models.repository;

import java.util.List;
import java.util.Set;

import org.springframework.data.neo4j.annotation.Query;
import org.springframework.data.neo4j.repository.Neo4jRepository;
import org.springframework.data.repository.query.Param;
import com.deepthought.models.Token;
import com.deepthought.models.edges.TokenWeight;

/**
 * Spring Data Repository pattern to perform CRUD operations and other various queries
 *  on {@link Token}
 */
public interface TokenRepository extends Neo4jRepository<Token, Long> {

	public Token findByValue(@Param("value") String value);

	/**
	 * Retrieves {@link Set} of {@link TokenWeight}s for a {@link Token} with a given value
	 *
	 * @param value value of the desired token for which to retrieve weights
	 *
	 * @return {@link Set} of {@link TokenWeights}
	 */
	@Query("MATCH (s:Token{value:{value}})-[fw:HAS_RELATED_TOKEN]->(t:Token) RETURN s, fw, t")
	public Set<TokenWeight> getTokenWeights(@Param("value") String value);

	/**
	 * Retrieves all {@link TokenWeight} connections between {@link Token} nodes with the provided values.
	 *
	 * @param input_value value of desired input token
	 * @param output_value value of desired output token
	 *
	 * @return {@link List} of {@link Token}s
	 */
	@Query("MATCH p=(f1:Token{value:{input_value}})-[fw:HAS_RELATED_TOKEN]->(f2:Token{value:{output_value}}) RETURN f1,fw,f2")
	public List<Token> getConnectedTokens(@Param("input_value") String input_value,
									      @Param("output_value") String output_value);

	/**
	 * Creates a {@linkplain TokenWeight weighted} connection between two tokens
	 *
	 * @param input_value
	 * @param output_value
	 * @param weight
	 */
	@Query("MATCH (f_in:Token),(f_out:Token)" +
			"WHERE f_in.value = {input_value} AND f_out.value = {output_value}" +
			"CREATE (f_in)-[r:HAS_RELATED_TOKEN{ weight: {weight}}]->(f_out)" +
			"RETURN r ")
	public TokenWeight createWeightedConnection(@Param("input_value") String input_value,
										        @Param("output_value") String output_value,
										        @Param("weight") double weight);
}
