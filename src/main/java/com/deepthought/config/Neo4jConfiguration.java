package com.deepthought.config;

import org.neo4j.ogm.session.SessionFactory;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.data.neo4j.Neo4jProperties;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.context.annotation.Configuration;
import org.springframework.data.neo4j.repository.config.EnableNeo4jRepositories;
import org.springframework.data.neo4j.transaction.Neo4jTransactionManager;
import org.springframework.data.neo4j.web.support.OpenSessionInViewInterceptor;
import org.springframework.transaction.annotation.EnableTransactionManagement;

@Configuration
@EnableNeo4jRepositories(basePackages = "com.deepthought.models.repository")
@EnableTransactionManagement
@ComponentScan({"com.deepthought.models"})
@EnableConfigurationProperties(Neo4jProperties.class)
public class Neo4jConfiguration {

	private final Neo4jProperties properties;

	public Neo4jConfiguration(Neo4jProperties properties) {
		this.properties = properties;
	}

	@Bean
	@ConditionalOnMissingBean
	public org.neo4j.ogm.config.Configuration configuration() {
		return this.properties.createConfiguration();
	}

	@Bean
	public OpenSessionInViewInterceptor openSessionInViewInterceptor() {
		OpenSessionInViewInterceptor openSessionInViewInterceptor =
				new OpenSessionInViewInterceptor();
		openSessionInViewInterceptor.setSessionFactory(sessionFactory(configuration()));
		return openSessionInViewInterceptor;
	}

	@Bean
	public SessionFactory sessionFactory(org.neo4j.ogm.config.Configuration configuration) {
		return new SessionFactory(configuration, "com.deepthought");
	}

	@Bean
	public Neo4jTransactionManager transactionManager(SessionFactory sessionFactory) {
		return new Neo4jTransactionManager(sessionFactory);
	}
}