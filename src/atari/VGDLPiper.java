package atari;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;

public class VGDLPiper {
	public static BufferedReader inp;
    public static BufferedWriter out;

	public static void main(String[] args) {
		try {
			doit();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public static String pipe(String msg) {
		String ret;

		try {
			out.write(msg + "\n");
			out.flush();
			ret = inp.readLine();
			return ret;
		}
		catch (Exception e) {}
		return "";
	}

	private static void doit() throws IOException {
//		String cmd = "python -m examples.gridphysics.aliens";
		String cmd = "c:\\Python27\\python";

		try {

			System.out.println(cmd);
			System.out.println(System.getProperty("user.dir"));
			Process p = Runtime.getRuntime().exec(cmd);

			inp = new BufferedReader( new InputStreamReader(p.getInputStream()) );
			out = new BufferedWriter( new OutputStreamWriter(p.getOutputStream()) );

//			System.out.println(pipe(" "));
			System.out.println(pipe("print 108"));
			System.out.println(pipe("exit()"));

			inp.close();
			out.close();
		}

		catch (Exception err) {
			err.printStackTrace();
		}
	}
}
