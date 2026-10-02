package bslsjdk.mcnpu;

import java.io.*;
import java.net.InetAddress;
import java.net.Socket;
import java.net.InetSocketAddress;

public final class NpuServiceClient {
    private static final int IPC_PORT = 38761;
    private NpuServiceClient() {}

    public static String request(String command) {
        if (command == null || command.isEmpty()) return "ERR EMPTY_COMMAND";
        try (Socket socket = new Socket()) {
            socket.connect(new InetSocketAddress(InetAddress.getLoopbackAddress(), IPC_PORT), 3000);
            socket.setSoTimeout(3000);
            BufferedWriter out = new BufferedWriter(new OutputStreamWriter(socket.getOutputStream()));
            BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
            out.write(command);
            out.write("\n");
            out.flush();
            String line = in.readLine();
            return line == null ? "ERR EMPTY_REPLY" : line;
        } catch (Throwable t) {
            return "ERR SERVICE_UNAVAILABLE " + t.getClass().getSimpleName() + " " + String.valueOf(t.getMessage());
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
