package bslsjdk.mcnpu;

import java.io.*;
import java.net.InetSocketAddress;
import java.net.Socket;

public final class NpuServiceClient {
    private static final int PORT = NpuService.PORT;
    private NpuServiceClient() {}

    public static String request(String command) {
        try (Socket socket = new Socket()) {
            socket.connect(new InetSocketAddress("127.0.0.1", PORT), 500);
            socket.setSoTimeout(3000);
            BufferedWriter out = new BufferedWriter(new OutputStreamWriter(socket.getOutputStream()));
            BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
            out.write(command);
            out.write("\n");
            out.flush();
            String line = in.readLine();
            return line == null ? "ERR EMPTY_REPLY" : line;
        } catch (Throwable t) {
            return "ERR SERVICE_UNAVAILABLE " + t.getClass().getSimpleName();
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
