package com.deepthought.observableStructs;

import java.util.Observable;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;


/**
 * A Node for a graph structure that is capable of connecting
 * to any number of input and output nodes.
 *
 * @param <T> Generic object
 * 
 * @threadsafe
 */
public class ConcurrentNode<T> extends Observable {
	private UUID uuid = null;
	private ConcurrentHashMap<UUID, ConcurrentNode<?>> inputs;
	private ConcurrentHashMap<UUID, ConcurrentNode<?>> outputs;
	private T data;
	private Class<?> clazz;
	volatile AtomicBoolean isEntryNode = new AtomicBoolean(false);
	
	public ConcurrentNode(T data){
		this.uuid = UUID.randomUUID();
		this.inputs = new ConcurrentHashMap<UUID, ConcurrentNode<?>>();
		this.outputs = new ConcurrentHashMap<UUID, ConcurrentNode<?>>();
		this.data = data;
		this.clazz = data.getClass();
	}
	
	public ConcurrentNode(
			ConcurrentHashMap<UUID, ConcurrentNode<?>> inputMap, 
			ConcurrentHashMap<UUID, ConcurrentNode<?>> outputMap, 
			T data)
	{
		this.uuid = UUID.randomUUID();
		this.inputs = inputMap;
		this.outputs = outputMap;
		this.data = data;
		this.clazz = data.getClass();
	}
	
	public ConcurrentNode() {
		this.uuid = UUID.randomUUID();
		
		this.inputs = new ConcurrentHashMap<UUID, ConcurrentNode<?>>();
		this.outputs = new ConcurrentHashMap<UUID, ConcurrentNode<?>>();
		this.data = null;
	}

	public void setData(T data){
		this.data = data;
	}
	
	public void addInput(UUID uuid, ConcurrentNode<?> node){
		//.0001 chosen for assumption of behaving as an extremely low weight for initial connection
		inputs.put(uuid, node);
		setChanged();
        notifyObservers();                                                                  // makes the observers print null
	}
	
	public ConcurrentNode<?> getInput(UUID uuid){
		return inputs.get(uuid);
	}
	
	public void addOutput(UUID uuid, ConcurrentNode<?> node){
		outputs.put(uuid, node);
		setChanged();
        notifyObservers(this);                                                                  // makes the observers print null
	}
	
	public ConcurrentNode<?> getOutput(ConcurrentNode<?> node){
		return outputs.get(node);
	}
	
	public ConcurrentHashMap<UUID, ConcurrentNode<?>> getOutputs(){
		return this.outputs;
	}
	
	public T getData(){
		return this.data;
	}
	
	public Class<? extends Object> getType(){
		return this.clazz;
	}
	
	/**
	 * 
	 * @return
	 */
	public UUID getUuid(){
		return uuid;
	}
}
