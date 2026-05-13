package com.deepthought;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.autoconfigure.data.neo4j.Neo4jDataAutoConfiguration;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.context.annotation.PropertySource;
import org.springframework.context.annotation.PropertySources;

@SpringBootApplication(exclude={Neo4jDataAutoConfiguration.class})
@ComponentScan(basePackages = {"com.deepthought"})
@PropertySources({
	@PropertySource("classpath:application.properties")
})
public class App 
{
    public static void main( String[] args )
    {
    	 SpringApplication.run(App.class, args);
	 }
}
