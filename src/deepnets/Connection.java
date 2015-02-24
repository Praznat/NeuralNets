package deepnets;

public class Connection {

	private final Node inputNode, outputNode;
	private AccruingWeight weight;
	
	public Connection(Node inputNode, Node outputNode, AccruingWeight weight) {
		this.inputNode = inputNode;
		this.outputNode = outputNode;
		this.weight = weight != null ? weight : new AccruingWeight();
	}

	public static <N extends Node> Connection quickCreate(N inputNode, N outputNode, AccruingWeight weight) {
		Connection conn = new Connection((Node) inputNode, (Node) outputNode, weight);
		inputNode.getOutputConnections().add(conn);
		outputNode.getInputConnections().add(conn);
		return conn;
	}
	public static <N extends Node> Connection getOrCreate(N inputNode, N outputNode) {
		return getOrCreate(inputNode, outputNode, null);
	}
	public static <N extends Node> Connection getOrCreate(N inputNode, N outputNode, AccruingWeight weight) {
		Connection conn = null;
		for (Connection outputConn : inputNode.getOutputConnections()) {
			if (outputConn.getOutputNode() == outputNode) {
				conn = outputConn;
				break;
			}
		}
		if (conn != null) { // just check input-output = output-input
			boolean isOK = false;
			for (Connection inputConn : outputNode.getInputConnections()) {
				if (inputConn.getInputNode() == inputNode && inputConn == conn) isOK = true;
			}
			if (!isOK) System.out.println("CRITICAL ERROR: input-output != output-input");
		}
		if (conn == null) conn = quickCreate(inputNode, outputNode, weight);
		if (inputNode == outputNode) System.out.println("CRITICAL ERROR: node connected to itself");
		return conn;
	}
	
	public AccruingWeight getWeight() {
		return weight;
	}

	public Node getInputNode() {
		return inputNode;
	}

	public Node getOutputNode() {
		return outputNode;
	}

	@Override
	public String toString() {
		return "C" + Utils.round(weight.getWeight(), 2);
	}
	
}
