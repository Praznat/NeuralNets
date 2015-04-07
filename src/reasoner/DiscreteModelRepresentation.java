package reasoner;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

public class DiscreteModelRepresentation {
	
	// TODO should be key:state, value:action+reward

	private Map<DiscreteState, Map<DiscreteState,Double>> map = new HashMap<DiscreteState,
			Map<DiscreteState,Double>>();
	private Map<DiscreteState, Integer> testMap = new HashMap<DiscreteState, Integer>();
	
	public static void main(String[] args) {
		testy();
	}

	private static void testy() {
		DiscreteModelRepresentation dmr = new DiscreteModelRepresentation();
		dmr.testMap.put(new DiscreteState(new double[] {0,1}), 1);
		dmr.testMap.put(new DiscreteState(new double[] {0,1}), 2);
		System.out.println(dmr.testMap.get(new DiscreteState(new double[] {0,1})));
	}
	
	public Double getTransitionProbability(DiscreteState s1, DiscreteState s2) {
		Map<DiscreteState,Double> transitions = map.get(s1);
		if (transitions == null) return null;
		Double p = transitions.get(s2);
		return p == null ? 0 : p;
	}

}
