package com.deepthought.brain;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Constructs {@linkplain Actions} experienced by the system
 *
 */
public class ActionFactory {
	private static Logger log = LoggerFactory.getLogger(ActionFactory.class);

	/*private static String[] actions = {"click",
								"doubleClick",
								"mouseover",
								"scroll",
								"sendKeys"};
	*/

	private static String[] actions = {"buy",
								"sell",
								"hold",
								"long",
								"short"};
	
	/**
	 * The list of actions possible
	 * @return
	 */
	public static String[] getActions(){
		return actions;
	}
}
