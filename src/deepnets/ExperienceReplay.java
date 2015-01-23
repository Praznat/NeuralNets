package deepnets;

import java.util.*;

public class ExperienceReplay {

	private final ArrayList<DataPoint> dataMemory = new ArrayList<DataPoint>();
	private final int maxSize;
	
	public ExperienceReplay(int maxSize) {
		this.maxSize = maxSize;
	}
	
	public void addMemory(DataPoint dp) {
		dataMemory.add(dp);
		if (dataMemory.size() > maxSize) dataMemory.remove(0);
	}

	public Collection<DataPoint> getBatch(int size, boolean resample) {
		Collection<DataPoint> result = new ArrayList<DataPoint>();
		Collection<DataPoint> replace = new ArrayList<DataPoint>();
		for (int i = 0; i < size; i++) {
			if (dataMemory.isEmpty()) return result;
			int k = (int) (dataMemory.size() * Math.random());
			DataPoint newDP = dataMemory.get(k);
			result.add(newDP);
			if (!resample) {
				dataMemory.remove(k);
				replace.add(newDP);
			}
		}
		dataMemory.addAll(replace);
		return result;
	}

	public Collection<DataPoint> getBatch(boolean resample) {
		return getBatch(dataMemory.size(), resample);
	}

}
