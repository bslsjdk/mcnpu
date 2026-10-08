package com.bslsjdk.mcnpu.next;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.os.Build;

/**
 * Restarts the NPU service without waiting for START_STICKY.
 *
 * A process killed by the LMK is not force-stopped, so an explicit broadcast still
 * reaches it and brings the process back. START_STICKY took 2m40s in a captured run;
 * this path is a normal app start.
 *
 * Trigger it manually with:
 *   adb shell am broadcast -a com.bslsjdk.mcnpu.next.action.WAKE -n com.bslsjdk.mcnpu.next/.NpuWakeReceiver
 */
public final class NpuWakeReceiver extends BroadcastReceiver {
    public static final String ACTION_WAKE = "com.bslsjdk.mcnpu.next.action.WAKE";

    @Override public void onReceive(Context ctx, Intent intent) {
        if (intent == null || !ACTION_WAKE.equals(intent.getAction())) return;
        android.util.Log.i("MCNPU", "WAKE received - starting service");
        try {
            Intent svc = new Intent(ctx, NpuService.class);
            if (Build.VERSION.SDK_INT >= 26) ctx.startForegroundService(svc);
            else ctx.startService(svc);
        } catch (Throwable t) {
            android.util.Log.e("MCNPU", "WAKE failed", t);
        }
    }
}
