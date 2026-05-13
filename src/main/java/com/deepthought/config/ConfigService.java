package com.deepthought.config;

import java.security.InvalidParameterException;
import java.text.MessageFormat;

public class ConfigService {
     
    public static String getProperty(String key){
    	String prop = System.getenv(key);
    	if(prop == null){
            throw new InvalidParameterException(MessageFormat.format("Missing value for key {0}!", key));
    	}
    	
    	return prop;
    }
}