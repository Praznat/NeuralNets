package reasoner;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

@SuppressWarnings("serial")
public class Forecast extends HashMap<DiscreteState,Double> {

	public Forecast(Map<DiscreteState, AtomicInteger> transitions, double cutoffProb) {
		super();
		int sum = 0;
		for (AtomicInteger ai : transitions.values()) sum += ai.get();
		for (DiscreteState ds : transitions.keySet()) {
			double p = transitions.get(ds).doubleValue() / sum;
			if (p > cutoffProb) this.put(ds, p);
		}
	}

}
