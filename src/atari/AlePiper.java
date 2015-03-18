package atari;

import java.io.*;
import java.util.*;

public class AlePiper {

	private static String ROM_DIR = "/Users/alexanderbraylan/fakeworkspace/Arcade-Learning-Environment/src/games/supported/";
	
	private static List<String> runString(int maxFrames, String game) {
		return java.util.Arrays.asList("./src/atari/run_ale.sh","-game_controller fifo", "-max_num_frames " + maxFrames,
				"/u/mhollen/sift/ale/roms/" + game + ".bin");
	}
	private static List<String> runStringAlt(int maxFrames, String game) {
		return java.util.Arrays.asList("./run_ale.sh","-game_controller","fifo","-max_num_frames",
				String.valueOf(maxFrames), "/u/mhollen/sift/ale/roms/" + game + ".bin");
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		try {
			doit();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public static void doit() throws IOException {
		String[] command = runStringAlt(100, "Asterix.cpp").toArray(new String[] {});
		ProcessBuilder builder = new ProcessBuilder(command);
		builder.directory(new File("ROM_DIR"));

		Process process = builder.start();

		InputStream is = process.getInputStream();
		InputStreamReader isr = new InputStreamReader(is);
		BufferedReader br = new BufferedReader(isr);
		String line;
		System.out.printf("Output of running %s is:\n",
				Arrays.toString(command));
		while ((line = br.readLine()) != null) {
			System.out.println(line);
		}

		//Wait to get exit value
		try {
			int exitValue = process.waitFor();
			System.out.println("\n\nExit Value is " + exitValue);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
	}

}
