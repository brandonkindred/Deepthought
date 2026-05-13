package com.qanairy.db;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.deepthought.models.Token;

/**
 * Defines static methods to handle the decomposition of it's data into their constituent pieces.
 */
public class DataDecomposer {
    private static Logger log = LoggerFactory.getLogger(DataDecomposer.class);

    /**
	 * Decomposes object into data fragments
	 *
	 * @return
	 * @throws IllegalArgumentException
	 * @throws IllegalAccessException
     * @throws JSONException
	 */
	public static List<Token> decompose(JSONObject jsonObject)
			throws IllegalArgumentException, IllegalAccessException, NullPointerException, JSONException{
		List<Token> objDefList = new ArrayList<Token>();
		Iterator<String> iter = jsonObject.keys();

		while(iter.hasNext()){
			String key = iter.next();
			log.info("Object Field Name :: "+key);
			Object value = jsonObject.get(key);
			if(value!=null){
	        	Token objDef = null;
	        	log.info("System class :: "+value.getClass());
	        	log.info("VALUE BEING READ :::  "+value.toString());
	        	if(value.toString().length()>=3 && value.toString().substring(0, 1).equals("{")){
	        		log.info("converting to json string");
	        		//String json_string = value.toString().substring(1, value.toString().length()-1);
		        	JSONObject obj = new JSONObject(value.toString());
		        	List<Token> definition_list = decompose(obj);
		        	objDefList.addAll(definition_list);
	        	}
	        	else if(value instanceof List){
		        	log.info("Deconstructing Array list");
		        	//return all elements of array
		        	List<Token> decomposedList = decomposeArrayList((List<?>)value);
	        		objDefList.addAll(decomposedList);
		        }
	        	else if(value.getClass().equals(JSONObject.class)){
	        		objDefList.addAll(decompose((JSONObject)value));
	        	}
		        else if(value.getClass().equals(String[].class)){
		        	log.info("Deconstructing String array");

		        	String[] array = (String[]) value;
		        	for(String stringVal : array){
		        		objDef = new Token(stringVal.toString());
		        		objDefList.add(objDef);
		            }
		        }
		        else if(value instanceof Object[]){
		        	log.info("Deconstructing Object list");

		        	Object[] array = (Object[]) value;
		        	List<Token> decomposedList = decomposeObjectArray(array);
		        	objDefList.addAll(decomposedList);
		        }
		        else if(value.getClass().equals(JSONArray.class)){
		        	log.info("Deconstructing JSONArray list");
		        	JSONArray array = (JSONArray) value;
		        	for(int idx=0; idx<array.length(); idx++){
		        		Object item = array.get(idx);
		        		if(item instanceof JSONObject){
		        			objDefList.addAll(decompose((JSONObject) item));
		        		}
		        		else if(item != null){
		        			objDefList.addAll(decompose(item.toString()));
		        		}
		        	}
		        }
		        else{
		        	String[] words = value.toString().split("\\s+");
		        	log.info("parsing string .....");
		        	for(String word : words){
		        		objDef = new Token(word);
		        		objDefList.add(objDef);
		        	}
		        }
	        	log.info("Ending list size :: "+objDefList.size());

			}
        }

		//log.info("Object definition List size :: "+objDefList.size());
		return objDefList;

	}

	/**
	 * Decomposes object into data fragments
	 *
	 * @return
	 * @throws IllegalArgumentException
	 * @throws IllegalAccessException
	 */
	public static List<Token> decompose(String value) throws IllegalArgumentException, IllegalAccessException, NullPointerException{
		List<Token> objDefList = new ArrayList<Token>();
    	String[] words = value.toString().split("\\s+");
    	for(String word : words){
    		log.info("word :: "+word);
    		Token objDef = new Token(word);
    		objDefList.add(objDef);
    	}

		return objDefList;
	}

	/**
	 * Decomposes object into data fragments
	 *
	 * @return
	 * @throws IllegalArgumentException
	 * @throws IllegalAccessException
	 */
	public static List<Token> decompose(Object obj) throws IllegalArgumentException, IllegalAccessException, NullPointerException{
		if(obj instanceof String){
			return decompose((String) obj);
		}
		if(obj instanceof List){
			return decomposeArrayList((List<?>) obj);
		}
		List<Token> objDefList = new ArrayList<Token>();
		Class<?> objClass = obj.getClass();
	    Field[] fields = objClass.getFields();
        log.info("LIST CLASS:: "+ objClass);
        log.info("FIELD COUNT : "+ fields.length);
	    for(Field field : fields) {
	        Object value = field.get(obj);
	        if(value!=null){
	        	Token objDef = null;

	        	if(value instanceof List){
		        	log.info("Deconstructing Array list");
		        	//return all elements of array
		        	List<Token> decomposedList = decomposeArrayList((List<?>)value);
	        		objDefList.addAll(decomposedList);
		        }
		        else if(value.getClass().equals(String[].class)){
		        	log.info("Deconstructing String array");

		        	String[] array = (String[]) value;
		        	for(String stringVal : array){
		        		objDef = new Token(stringVal.toString());
		        		objDefList.add(objDef);
		            }
		        }
		        else if(value instanceof Object[]){
		        	log.info("Deconstructing Object list");

		        	Object[] array = (Object[]) value;
		        	List<Token> decomposedList = decomposeObjectArray(array);
		        	objDefList.addAll(decomposedList);
		        }
		        else{
		        	log.info("Creating object definition for field");
		        	String[] words = value.toString().split("\\s+");
		        	log.info("VALUE :: " +value.toString());
		        	log.info("words :: "+words.length);
		        	for(String word : words){
		        		log.info("word :: "+word);
		        		objDef = new Token(word);
		        		objDefList.add(objDef);
		        	}
		        }
	        }
	    }
		return objDefList;
	}

	/**
	 * Decomposes object into data fragments
	 *
	 * @return
	 * @throws IllegalArgumentException
	 * @throws IllegalAccessException
	 */
	public static List<Token> decompose(HashMap<?,?> map) throws IllegalArgumentException, IllegalAccessException, NullPointerException{
		List<Token> objDefList = new ArrayList<Token>();

		Class<?> objClass = map.getClass();
        log.info("LIST CLASS:: "+ objClass);

		for(Object key : map.keySet()){
			Object value = map.get(key);
			if(value!=null){
	        	Token objDef = null;

	        	if(value instanceof List){
		        	log.info("Deconstructing Array list");
		        	//return all elements of array
		        	List<Token> decomposedList = decomposeArrayList((List<?>)value);
	        		objDefList.addAll(decomposedList);
		        }
		        else if(value.getClass().equals(String[].class)){
		        	log.info("Deconstructing String array");

		        	String[] array = (String[]) value;
		        	for(String stringVal : array){
		        		objDef = new Token(stringVal.toString());
		        		objDefList.add(objDef);
		            }
		        }
		        else if(value instanceof Object[]){
		        	log.info("Deconstructing Object list");

		        	Object[] array = (Object[]) value;
		        	List<Token> decomposedList = decomposeObjectArray(array);
		        	objDefList.addAll(decomposedList);
		        }
		        else{
		        	log.info("Creating object definition for field");
		        	String[] words = value.toString().split("\\s+");

		        	for(String word : words){
		        		objDef = new Token(word);
		        		objDefList.add(objDef);
		        	}
		        }
	        }
		}
		return objDefList;
	}

	/**
	 * Decomposes an array of Objects into memory blocks
	 * @param array
	 * @return
	 * @throws IllegalArgumentException
	 * @throws IllegalAccessException
	 */
	private static List<Token> decomposeObjectArray(Object[] array) throws IllegalArgumentException, IllegalAccessException{
    	List<Token> objDefList = new ArrayList<Token>();
		if(array == null || array.length == 0){
			return objDefList;
		}

        for(Object object : array){
        	objDefList.addAll(DataDecomposer.decompose(object));
        }
		return objDefList;
	}

	/**
	 * Iterates over ArrayList of objects, and decomposes each object
	 *
	 * @param list
	 * @return
	 * @throws IllegalArgumentException
	 * @throws IllegalAccessException
	 */
	private static List<Token> decomposeArrayList(List<?> list) throws IllegalArgumentException, IllegalAccessException, NullPointerException {
    	List<Token> objDefList = new ArrayList<Token>();
		if(list == null || list.isEmpty()){
			return objDefList;
		}

        for(Object object : list){
        	if(object != null){
        		objDefList.addAll(DataDecomposer.decompose(object));
        	}
        }
		return objDefList;
	}
}
