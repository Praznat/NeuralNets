package sparsenets;

import java.util.*;

import deepnets.*;


public class SparseNode extends Node {

	private static final Comparator<Connection> CONN_COMP = new Comparator<Connection>() {
		@Override
		public int compare(Connection o1, Connection o2) {
			return Math.abs(o1.getWeight().getWeight()) < Math.abs(o2.getWeight().getWeight()) ? 1 : -1;
		}
	};
	
	private final int numUsedConnections;
	
	public SparseNode(ActivationFunction activationFunction, Layer<? extends Node> parentLayer,
			String nodeInLayer, int numUsedConnections) {
		super(activationFunction, parentLayer, nodeInLayer);
		this.numUsedConnections = numUsedConnections;
		inputConnections = new TreeSet<Connection>();
		outputConnections = new TreeSet<Connection>();
	}
	
	private Collection<Connection> getMaxConnections(Collection<Connection> conns) {
		Collection<Connection> result = new ArrayList<Connection>();
		Iterator<Connection> iter = conns.iterator();
		for (int i = 0; i < numUsedConnections; i++) {
			if (iter.hasNext()) result.add(iter.next());
			else break;
		}
		return result;
	}
	
	@Override
	public Collection<Connection> getInputConnections() {
		Collection<Connection> result = getMaxConnections(inputConnections);
		result.add(BiasNode.getBiasConnection(this));
		return result;
	}

	@Override
	public Collection<Connection> getOutputConnections() {
		return getMaxConnections(outputConnections);
	}
}

