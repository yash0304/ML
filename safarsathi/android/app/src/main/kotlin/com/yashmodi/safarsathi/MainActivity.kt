package com.yashmodi.safarsathi

import android.app.Activity
import android.content.ActivityNotFoundException
import android.content.Intent
import android.provider.ContactsContract
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel

/**
 * Picks ONE number from the phone's own contacts app.
 *
 * NO CONTACTS PERMISSION. ACTION_PICK on the Phone table hands the system
 * picker to the user; they choose one number, and this app receives a
 * one-time grant to read that single row. It never sees the address book —
 * which is the only honest way for an app whose whole claim is "nothing
 * leaves this phone, nothing is read that you did not hand it" to do this.
 * READ_CONTACTS would read everyone, and the manifest does not ask for it.
 */
class MainActivity : FlutterActivity() {
    private var pending: MethodChannel.Result? = null

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL)
            .setMethodCallHandler { call, result ->
                when (call.method) {
                    "pickPhone" -> pickPhone(result)
                    else -> result.notImplemented()
                }
            }
    }

    private fun pickPhone(result: MethodChannel.Result) {
        if (pending != null) {
            result.error("busy", "The contact picker is already open.", null)
            return
        }
        val intent = Intent(
            Intent.ACTION_PICK,
            ContactsContract.CommonDataKinds.Phone.CONTENT_URI,
        )
        pending = result
        try {
            startActivityForResult(intent, PICK_PHONE)
        } catch (e: ActivityNotFoundException) {
            pending = null
            result.error(
                "unavailable",
                "This phone has no contacts app to pick from.",
                null,
            )
        }
    }

    @Deprecated("FlutterActivity is not a ComponentActivity; this is its result path.")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode != PICK_PHONE) return
        val result = pending ?: return
        pending = null

        val uri = data?.data
        if (resultCode != Activity.RESULT_OK || uri == null) {
            // Backed out of the picker. Not an error: nothing was chosen.
            result.success(null)
            return
        }

        try {
            val cursor = contentResolver.query(
                uri,
                arrayOf(
                    ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME,
                    ContactsContract.CommonDataKinds.Phone.NUMBER,
                ),
                null,
                null,
                null,
            )
            if (cursor == null) {
                result.success(null)
                return
            }
            cursor.use {
                if (it.moveToFirst()) {
                    result.success(
                        mapOf(
                            "name" to (it.getString(0) ?: ""),
                            "number" to (it.getString(1) ?: ""),
                        ),
                    )
                } else {
                    result.success(null)
                }
            }
        } catch (e: Exception) {
            result.error("read", "That contact could not be read.", null)
        }
    }

    companion object {
        private const val CHANNEL = "safarsathi/contacts"
        private const val PICK_PHONE = 4107
    }
}
