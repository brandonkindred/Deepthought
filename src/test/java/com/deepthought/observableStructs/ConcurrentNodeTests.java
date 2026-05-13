package com.deepthought.observableStructs;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertNotNull;
import static org.testng.Assert.assertNull;
import static org.testng.Assert.assertSame;
import static org.testng.Assert.assertTrue;

import java.util.Observable;
import java.util.Observer;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

import org.testng.annotations.Test;

@Test(groups = "Regression")
public class ConcurrentNodeTests {

    @Test
    public void constructorsInitializeDataAndMaps() {
        ConcurrentNode<String> node = new ConcurrentNode<>("value");
        assertEquals(node.getData(), "value");
        assertEquals(node.getType(), String.class);
        assertNotNull(node.getUuid());
        assertTrue(node.getOutputs().isEmpty());

        ConcurrentNode<String> emptyNode = new ConcurrentNode<>();
        assertNull(emptyNode.getData());
        assertNotNull(emptyNode.getUuid());

        ConcurrentHashMap<UUID, ConcurrentNode<?>> inputs = new ConcurrentHashMap<>();
        ConcurrentHashMap<UUID, ConcurrentNode<?>> outputs = new ConcurrentHashMap<>();
        ConcurrentNode<String> mapCtorNode = new ConcurrentNode<>(inputs, outputs, "x");
        assertSame(mapCtorNode.getOutputs(), outputs);
    }

    @Test
    public void addInputAndOutput_storeConnectionsAndNotify() {
        ConcurrentNode<String> target = new ConcurrentNode<>("target");
        ConcurrentNode<String> source = new ConcurrentNode<>("source");
        ConcurrentNode<String> destination = new ConcurrentNode<>("destination");

        AtomicInteger notifications = new AtomicInteger();
        target.addObserver(new Observer() {
            @Override
            public void update(Observable o, Object arg) {
                notifications.incrementAndGet();
            }
        });

        UUID inputId = UUID.randomUUID();
        UUID outputId = UUID.randomUUID();

        target.addInput(inputId, source);
        target.addOutput(outputId, destination);

        assertSame(target.getInput(inputId), source);
        assertEquals(target.getOutputs().get(outputId), destination);
        assertEquals(notifications.get(), 2);
    }
}
