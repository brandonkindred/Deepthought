package com.deepthought.observableStructs;

import static org.testng.Assert.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Observable;
import java.util.Observer;
import java.util.UUID;
import java.util.concurrent.ConcurrentLinkedQueue;

import org.testng.annotations.Test;

@Test(groups = "Regression")
public class ObservableStructsTests {

    @Test
    public void observableQueue_mutationsNotifyObserversAndMutateState() {
        ObservableQueue<String> queue = new ObservableQueue<>();
        List<Object> updates = new ArrayList<>();
        Observer observer = (Observable o, Object arg) -> updates.add(arg);
        queue.addObserver(observer);

        assertTrue(queue.add("a"));
        assertTrue(queue.offer("b"));
        assertEquals(queue.size(), 2);
        assertEquals(queue.peek(), "a");

        assertEquals(queue.poll(), "a");
        assertEquals(queue.remove(), "b");
        assertTrue(queue.isEmpty());

        queue.addAll(Arrays.asList("x", "y"));
        assertTrue(queue.contains("x"));
        assertTrue(queue.remove("x"));
        queue.clear();

        assertTrue(updates.size() >= 7, "Expected observer notifications from queue operations");
    }

    @Test
    public void observableHash_putAndRandomKeyWorkAsExpected() {
        ObservableHash<String, Integer> hash = new ObservableHash<>();
        List<Object> updates = new ArrayList<>();
        hash.addObserver((o, arg) -> updates.add(arg));

        ConcurrentLinkedQueue<Integer> first = hash.put("numbers", 1);
        hash.put("numbers", 2);
        assertEquals(first.size(), 2);

        ConcurrentLinkedQueue<Integer> replacement = new ConcurrentLinkedQueue<>();
        replacement.add(9);
        hash.put("other", replacement);

        assertEquals(hash.size(), 2);
        assertFalse(hash.isEmpty());
        assertEquals(hash.getQueueHash().get("other").peek().intValue(), 9);

        Object randomKey = hash.getRandomKey();
        assertTrue(hash.getQueueHash().containsKey(randomKey));
        assertEquals(updates.size(), 3);
    }

    @Test
    public void concurrentNode_tracksConnectionsAndData() {
        ConcurrentNode<String> root = new ConcurrentNode<>("root");
        ConcurrentNode<String> child = new ConcurrentNode<>("child");
        UUID childId = child.getUuid();

        List<Object> updates = new ArrayList<>();
        root.addObserver((o, arg) -> updates.add(arg));

        root.addInput(childId, child);
        root.addOutput(childId, child);

        assertEquals(root.getInput(childId), child);
        assertEquals(root.getOutputs().get(childId), child);
        assertEquals(root.getData(), "root");
        assertEquals(root.getType(), String.class);

        root.setData("newRoot");
        assertEquals(root.getData(), "newRoot");
        assertEquals(updates.size(), 2);

        ConcurrentNode<Object> empty = new ConcurrentNode<>();
        assertNotNull(empty.getUuid());
        assertNull(empty.getData());
    }
}
