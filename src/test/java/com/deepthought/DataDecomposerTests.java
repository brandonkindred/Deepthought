package com.deepthought;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertTrue;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import com.deepthought.resourceClasses.SelfContainedTestObject;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.testng.annotations.Test;

import com.deepthought.models.Token;
import com.deepthought.db.DataDecomposer;

@Test(groups = "Regression")
public class DataDecomposerTests {

	@Test
	public void decomposeGenericObject() throws IllegalArgumentException, IllegalAccessException, JSONException {
		String[] keys = { "String", "value", "object", "key", "here" };
		JSONObject jsonObj = new JSONObject();
		jsonObj.put("string_val", "String value");
		JSONObject obj = new JSONObject();
		obj.put("object_key", "object key here");
		jsonObj.put("obj", obj);

		List<Token> objectDefinitionList = DataDecomposer.decompose(jsonObj);
		Map<String, Token> map = new HashMap<String, Token>();
		for (Token token : objectDefinitionList) {
			map.put(token.getValue(), token);
		}

		for (String key : keys) {
			map.remove(key);
		}

		assertTrue(map.isEmpty());
	}

	@Test
	public void decomposeMap() throws IllegalArgumentException, IllegalAccessException {
		HashMap<String, Object> map = new HashMap<String, Object>();
		map.put("title", "hello world");
		map.put("labels", Arrays.asList("alpha beta", null));
		map.put("tokens", new String[] { "foo", "bar" });
		map.put("unused", null);

		List<Token> output = DataDecomposer.decompose(map);
		List<String> values = output.stream().map(Token::getValue).collect(Collectors.toList());

		assertTrue(values.contains("hello"));
		assertTrue(values.contains("world"));
		assertTrue(values.contains("alpha"));
		assertTrue(values.contains("beta"));
		assertTrue(values.contains("foo"));
		assertTrue(values.contains("bar"));
	}

	@Test
	public void decomposeObjectList() throws IllegalArgumentException, IllegalAccessException {
		SelfContainedTestObject input = new SelfContainedTestObject(null, 0, null, 0.0, null, null);
		input.string_value = "quick brown fox";
		input.int_primitive_value = 7;
		input.string_list = Arrays.asList("jumps high", "over");

		List<Token> output = DataDecomposer.decompose((Object) input);
		List<String> values = output.stream().map(Token::getValue).collect(Collectors.toList());

		assertTrue(values.contains("quick"));
		assertTrue(values.contains("brown"));
		assertTrue(values.contains("fox"));
		assertTrue(values.contains("jumps"));
		assertTrue(values.contains("high"));
	}

	@Test
	public void decomposeObjectArray() throws IllegalArgumentException, IllegalAccessException, JSONException {
		JSONObject jsonObj = new JSONObject();
		jsonObj.put("items", new Object[] { "alpha beta", "gamma" });
		jsonObj.put("nested", new JSONArray(Arrays.asList(new JSONObject().put("k", "delta"))));

		List<Token> output = DataDecomposer.decompose(jsonObj);
		List<String> values = output.stream().map(Token::getValue).collect(Collectors.toList());

		assertEquals(values.stream().filter("alpha"::equals).count(), 1L);
		assertTrue(values.contains("beta"));
		assertTrue(values.contains("gamma"));
		assertTrue(values.contains("delta"));
	}
}
