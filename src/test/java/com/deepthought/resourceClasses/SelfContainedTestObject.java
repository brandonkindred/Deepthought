package com.deepthought.resourceClasses;

import java.util.List;
import java.util.Map;

public class SelfContainedTestObject {
	public String string_value;
	public int	  int_primitive_value;
	public Integer int_object_value;
	public double  double_value;
	public Double  double_object_value;
	public float   float_primitive_value;
	
	public List<String> string_list;
	public Map<String, String> string_map;
	
	public SelfContainedTestObject(String string_value, int primitive_int, Integer int_object, double double_primitive, List<String> string_list, Map<String, String> string_map){
		
	}
}
