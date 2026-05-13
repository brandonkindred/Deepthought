package com.deepthought.observableStructs;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertNotNull;
import static org.testng.Assert.assertTrue;

import java.util.Arrays;
import java.util.HashSet;
import java.util.concurrent.ConcurrentLinkedQueue;

import org.testng.annotations.Test;

@Test(groups = "Regression")
public class ObservableHashTests {

    @Test
    public void putSingleValues_createsAndAppendsQueues() {
        ObservableHash<String, Integer> hash = new ObservableHash<>();

        ConcurrentLinkedQueue<Integer> first = hash.put("k1", 1);
        ConcurrentLinkedQueue<Integer> second = hash.put("k1", 2);

        assertNotNull(first);
        assertEquals(hash.size(), 1);
        assertEquals(second.size(), 2);
        assertTrue(second.containsAll(Arrays.asList(1, 2)));
    }

    @Test
    public void putQueueAndGetRandomKey_returnsExistingKey() {
        ObservableHash<String, Integer> hash = new ObservableHash<>();
        ConcurrentLinkedQueue<Integer> queue = new ConcurrentLinkedQueue<>();
        queue.add(10);

        hash.put("alpha", queue);
        hash.put("beta", 20);

        Object randomKey = hash.getRandomKey();
        assertNotNull(randomKey);
        assertTrue(new HashSet<>(Arrays.asList("alpha", "beta")).contains(randomKey));
    }
}
