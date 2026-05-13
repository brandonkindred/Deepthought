package com.deepthought.observableStructs;

import java.util.HashMap;
import java.util.Observable;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ConcurrentLinkedQueue;

/**
 * Creates an Hash that is observable
 *
 * @param <E>
 */
public class ObservableHash<K, E> extends Observable {
	
	HashMap<K, ConcurrentLinkedQueue<E>> queueHash = null;
	
	/**
	 * 
	 */
	public ObservableHash() {
		queueHash = new HashMap<K, ConcurrentLinkedQueue<E>>();
	}

	public int size() {
		return queueHash.size();
	}

	public boolean isEmpty() {
		return queueHash.isEmpty();
	}

	public HashMap<K, ConcurrentLinkedQueue<E>> getQueueHash(){
		return this.queueHash;
	}
	
	/**
	 *	{@inheritDoc}
	 */
	public ConcurrentLinkedQueue<E> put(K key, E value) {
		System.out.println(Thread.currentThread().getName()  + " -> Adding queue to hash with key "+key);
		ConcurrentLinkedQueue<E> queue = queueHash.get(key);
		setChanged();
		if(queue == null){
			queue = new ConcurrentLinkedQueue<E>();
			queue.add(value);
			queueHash.put(key, queue);
		}
		else {
			queue.add(value);
			queueHash.put(key, queue);
		}	
		notifyObservers(key);
		return queue;
	}
	
	/**
	 *	{@inheritDoc}
	 */
	public ConcurrentLinkedQueue<E> put(K key, ConcurrentLinkedQueue<E> queue) {
		System.out.println(Thread.currentThread().getName()  + " -> Adding queue to hash with key "+key);
		setChanged();
		queueHash.put(key, queue);

		notifyObservers(key);
		return queue;
	}

	/**
	 * Finds random key that exists in hash
	 * @return <= 99999
	 */
	public synchronized Object getRandomKey(){
		Set<?> keys = queueHash.keySet();
		int total_keys = keys.size();
		Random rand = new Random();
		int rand_idx = rand.nextInt(total_keys);
		Object key_object = null;
		int key_idx = 0;
		for(Object key : keys){
			if(key_idx == rand_idx){
				key_object = key;
				break;
			}

			key_idx++;
		}
		
		return key_object;
	}
}
