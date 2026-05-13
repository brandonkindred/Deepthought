package com.deepthought.observableStructs;

import java.util.Collection;
import java.util.Iterator;
import java.util.Observable;
import java.util.Queue;
import java.util.concurrent.LinkedBlockingQueue;

/**
 *
 * @param <E>
 */
public class ObservableQueue<E> extends Observable implements Queue<E>{

	private Queue<E> queue;
	
	public ObservableQueue() {
		queue = new LinkedBlockingQueue<E>();
	}
	
	/**
	 * {@inheritDoc}
	 */
	public E poll(){
		setChanged();
		E element = queue.poll();
		if(element!= null){
			notifyObservers();
		}
		return element;
	}
	
	/**
	 * {@inheritDoc}
	 */
	public boolean add(E o) throws IllegalStateException{
		setChanged();
		boolean wasAdded = this.queue.add(o);
		notifyObservers(o);
		return wasAdded;
	}

	/**
	 * {@inheritDoc}
	 */
	public int size() {
		return queue.size();
	}

	/**
	 * {@inheritDoc}
	 */
	public boolean isEmpty() {
		return queue.isEmpty();
	}
	
	/**
	 * {@inheritDoc}
	 */
	public boolean contains(Object o) {
		return queue.contains(o);
	}

	/**
	 * {@inheritDoc}
	 */
	public Iterator<E> iterator() {
		return queue.iterator();
	}

	/**
	 * {@inheritDoc}
	 */
	public Object[] toArray() {
		return queue.toArray();
	}

	/**
	 * {@inheritDoc}
	 */
	public <T> T[] toArray(T[] a) {
		return queue.toArray(a);
	}

	/**
	 * {@inheritDoc}
	 */
	public boolean remove(Object o) {
		setChanged();
		boolean removed =  queue.remove(o);
		notifyObservers();
		return removed;
	}

	/**
	 * {@inheritDoc}
	 */
	public boolean containsAll(Collection<?> c) {
		return queue.containsAll(c);
	}

	/**
	 * {@inheritDoc}
	 */
	public boolean addAll(Collection<? extends E> c) {
		setChanged();
		boolean allAdded = queue.addAll(c);
		notifyObservers();
		return allAdded;
	}

	/**
	 * {@inheritDoc}
	 */
	public boolean removeAll(Collection<?> c) {
		setChanged();
		boolean removed = queue.removeAll(c);
        notifyObservers(this);
		return removed;
	}

	/**
	 * {@inheritDoc}
	 */
	public boolean retainAll(Collection<?> c) {
		setChanged();
		boolean retained = queue.retainAll(c);
        notifyObservers(this);
		return retained;
	}

	/**
	 * {@inheritDoc}
	 */
	public void clear() {
		setChanged();
		queue.clear();
		notifyObservers();
	}

	/**
	 * {@inheritDoc}
	 */
	public boolean offer(E e) {
		setChanged();
		boolean offerAccepted = queue.offer(e);
		System.out.println("OFFERING ELEMENT...");
        notifyObservers();
		return offerAccepted;
	}

	
	/**
	 * {@inheritDoc}
	 */
	public E remove() {
		setChanged();
		E element = queue.remove();
		notifyObservers();
		return element;
	}

	/**
	 * {@inheritDoc}
	 */
	public E element() {
		return queue.element();
	}

	/**
	 * {@inheritDoc}
	 */
	public E peek() {
		return queue.peek();
	}
	

}