package deepnets.indirectencodings;

import deepnets.*;

public class IndirectWeight implements Weight {
	private final CPPN cppn;
	private final Connection parentConnection;
	
	public IndirectWeight(CPPN cppn, Connection parentConnection) {
		this.cppn = cppn;
		this.parentConnection = parentConnection;
	}
	
	@Override
	public double getWeight() {
		return cppn.calculateConnectionWeight(parentConnection);
	}
	
}
