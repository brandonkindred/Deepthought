package com.deepthought.models.repository;

import org.springframework.data.neo4j.repository.Neo4jRepository;

import com.deepthought.models.LogisticRegressionModel;

public interface LogisticRegressionModelRepository extends Neo4jRepository<LogisticRegressionModel, Long> {
}
