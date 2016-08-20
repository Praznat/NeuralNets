package ann;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

public class BiasNode extends Node {
	
	private static final long serialVersionUID = -8176764910268356606L;
	//	public static final Node INSTANCE = ((ArrayList<Node>)Layer.createInputLayer(1, Node.BASIC_NODE_FACTORY).getNodes()).get(0);
	private static final String BIASNAME = "BIAS";
	private final Map<Node, Connection> toNodesMap = new HashMap<Node, Connection>();

	private BiasNode() {
		super(ActivationFunction.LINEAR, Layer.createInputLayer(1, Node.BASIC_NODE_FACTORY), BIASNAME);
	}

	public static BiasNode create() {
		BiasNode result = new BiasNode();
		result.clamp(1);
		result.setName(BIASNAME);
		return result;
	}
	
	public void connectToLayer(Layer<? extends Node> outputLayer) {
		for (Node n : outputLayer.getNodes()) connectToNode(n);
	}

	public void connectToNode(Node node) {
		connectToNode(node, null);
	}
	public void connectToNode(Node node, AccruingWeight weight) {
		Connection conn = weight == null ? Connection.getOrCreate(this, node)
				: Connection.getOrCreate(this, node, weight);
		toNodesMap.put(node, conn);
		if (toNodesMap.size() > 10000) {
			System.out.println("MEMORY LEAK WARNING! Clear biases please!");
		}
	}
	
//	public static void clearConnections() {
//		INSTANCE.getOutputConnections().clear();
//		toNodesMap.clear();
//	}
	
	public void disconnectFrom(Node n) {
		for (Iterator<Connection> iter = this.getOutputConnections().iterator(); iter.hasNext();) {
			Connection conn = iter.next();
			if (conn.getOutputNode() == n) iter.remove();
		}
	}
	
	public Connection getBiasConnection(Node n) {
		return toNodesMap.get(n);
	}

	public static boolean isBias(Node hNode) {
		return BIASNAME.equals(hNode.toString());
	}
	
}
