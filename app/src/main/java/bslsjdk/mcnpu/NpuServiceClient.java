package bslsjdk.mcnpu;

import java.io.*;
import java.net.InetAddress;
import java.net.Socket;
import java.net.InetSocketAddress;

public final class NpuServiceClient {
    private static final int IPC_PORT = 38761;
    /** Must match NpuService, which binds explicitly to 127.0.0.1. */
    private static final String IPC_HOST = "127.0.0.1";
    private static final int CONNECT_TIMEOUT_MS = 3000;
    private static final int READ_TIMEOUT_MS = 15000;
    private NpuServiceClient() {}

    public static String request(String command) {
        if (command == null || command.isEmpty()) return "ERR EMPTY_COMMAND";
        long t0 = System.nanoTime();
        try (Socket socket = new Socket()) {
            socket.connect(new InetSocketAddress(InetAddress.getByName(IPC_HOST), IPC_PORT), CONNECT_TIMEOUT_MS);
            // Long enough for a cold graph build on the largest shape: the first
            // n=16384 ADD pays graph creation, which measured ~150-230 ms, and a 3 s
            // ceiling turned a slow-but-valid first case into SERVICE_UNAVAILABLE.
            socket.setSoTimeout(READ_TIMEOUT_MS);
            BufferedWriter out = new BufferedWriter(new OutputStreamWriter(socket.getOutputStream()), 64 * 1024);
            BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()), 64 * 1024);
            out.write(command);
            out.write("\n");
            out.flush();
            String line = in.readLine();
            return line == null ? "ERR EMPTY_REPLY" : line;
        } catch (Throwable t) {
            // Keep the concrete exception type: it is the only way to tell
            // refused (not listening) from timeout (blocked) from reset.
            return "ERR SERVICE_UNAVAILABLE " + t.getClass().getName()
                    + " msg=" + String.valueOf(t.getMessage())
                    + " us=" + ((System.nanoTime() - t0) / 1000);
        }
    }

    public static boolean isAvailable() {
        return request("PING").startsWith("PONG MCNPU/");
    }

    public static String status() {
        return request("STATUS");
    }

    public static String smoke() {
        return request("SMOKE");
    }
}
