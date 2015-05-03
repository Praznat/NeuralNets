package deepnets;

import java.util.*;

public class ExperienceReplay<T extends ExperienceReplay.Memory> {

	private final ArrayList<T> dataMemory = new ArrayList<T>();
	private final int maxSize;

	public ExperienceReplay(int maxSize) {
		this(maxSize, false);
	}
	// TODO overwrite duplicate experiences!
	public ExperienceReplay(int maxSize, boolean overwriteCopies) {
		this.maxSize = maxSize;
	}
	
	public void addMemory(T dp) {
		dataMemory.add(dp);
		if (dataMemory.size() > maxSize) dataMemory.remove(0);
	}

	public Collection<T> getBatch(int size, boolean resample) {
		Collection<T> result = new ArrayList<T>();
		Collection<T> replace = new ArrayList<T>();
		for (int i = 0; i < size; i++) {
			if (dataMemory.isEmpty()) return result;
			int k = (int) (dataMemory.size() * Math.random());
			T newDP = dataMemory.get(k);
			result.add(newDP);
			if (!resample) {
				dataMemory.remove(k);
				replace.add(newDP);
			}
		}
		dataMemory.addAll(replace);
		return result;
	}

	public Collection<T> getBatch() {
		return getBatch(dataMemory.size(), false);
	}
	
	public boolean isFull() {
		return dataMemory.size() == maxSize;
	}

	public static interface Memory {
	}

	public void clear() {
		dataMemory.clear();
	}
	public double getSize() {
		return dataMemory.size();
	}
}
