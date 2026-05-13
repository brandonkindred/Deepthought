package com.deepthought.observableStructs;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertNull;
import static org.testng.Assert.assertTrue;

import java.util.Arrays;
import java.util.Observable;
import java.util.Observer;
import java.util.concurrent.atomic.AtomicInteger;

import org.testng.annotations.Test;

@Test(groups = "Regression")
public class ObservableQueueTests {

    @Test
    public void addAndPoll_updatesSizeAndNotifiesObservers() {
        ObservableQueue<String> queue = new ObservableQueue<>();
        AtomicInteger notifications = new AtomicInteger(0);
        Observer observer = new Observer() {
            @Override
            public void update(Observable o, Object arg) {
                notifications.incrementAndGet();
                if (notifications.get() == 1) {
                    assertEquals(arg, "first");
                }
            }
        };
        queue.addObserver(observer);

        assertTrue(queue.add("first"));
        assertEquals(queue.size(), 1);

        assertEquals(queue.poll(), "first");
        assertEquals(queue.size(), 0);
        assertEquals(notifications.get(), 2);

        assertNull(queue.poll());
        assertEquals(notifications.get(), 2, "No notification when polling empty queue");
    }

    @Test
    public void bulkMutations_changeContentsAsExpected() {
        ObservableQueue<Integer> queue = new ObservableQueue<>();

        assertTrue(queue.addAll(Arrays.asList(1, 2, 3, 4)));
        assertEquals(queue.size(), 4);
        assertTrue(queue.containsAll(Arrays.asList(1, 2, 3, 4)));

        assertTrue(queue.removeAll(Arrays.asList(2, 4)));
        assertFalse(queue.contains(2));
        assertFalse(queue.contains(4));

        assertTrue(queue.retainAll(Arrays.asList(1)));
        assertEquals(queue.size(), 1);
        assertTrue(queue.contains(1));

        queue.clear();
        assertTrue(queue.isEmpty());
    }
}
