import java.io.FileInputStream
import java.util.Properties

// THE SIGNING KEY DECIDES WHETHER AN UPDATE INSTALLS AT ALL.
//
// Until this was here, the release build was signed with `signingConfigs.debug`
// — the Flutter template's default. On a developer's own machine that is
// merely untidy: the debug keystore is generated once and reused, so upgrades
// work. On a CI runner there is no `~/.android/debug.keystore`, so Gradle
// makes a NEW RANDOM ONE on every single run, and Android refuses to upgrade
// an app whose signature changed. Every build produced a package that could
// only be installed by first uninstalling the last one, which wipes the
// database — every contact, every expense, the whole trip.
//
// `android/key.properties` is gitignored and holds the real key. Without it
// the build still works and still falls back to debug signing, because a
// contributor without the keystore should not be blocked; the log says which
// one was used. See docs/safarsathi/RUNNING.md.
val keystorePropertiesFile = rootProject.file("key.properties")
val keystoreProperties = Properties().apply {
    if (keystorePropertiesFile.exists()) {
        load(FileInputStream(keystorePropertiesFile))
    }
}
val hasReleaseKey = keystorePropertiesFile.exists()

plugins {
    id("com.android.application")
    // The Flutter Gradle Plugin must be applied after the Android and Kotlin Gradle plugins.
    id("dev.flutter.flutter-gradle-plugin")
}

android {
    namespace = "com.yashmodi.safarsathi"
    compileSdk = flutter.compileSdkVersion
    ndkVersion = flutter.ndkVersion

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    defaultConfig {
        // TODO: Specify your own unique Application ID (https://developer.android.com/studio/build/application-id.html).
        applicationId = "com.yashmodi.safarsathi"
        // You can update the following values to match your application needs.
        // For more information, see: https://flutter.dev/to/review-gradle-config.
        minSdk = flutter.minSdkVersion
        targetSdk = flutter.targetSdkVersion
        // Uses the version code from pubspec.yaml. When using split APKs, 1000 * ABI_VERSION
        // is added automatically by Flutter. (https://developer.android.com/studio/build/configure-apk-splits#configure-APK-versions)
        // You can force using the value of versionCode by specifying the `-P force-version-code-ignoring-abi=true`
        // flag during build.
        versionCode = flutter.versionCode
        versionName = flutter.versionName
    }

    signingConfigs {
        if (hasReleaseKey) {
            create("release") {
                keyAlias = keystoreProperties["keyAlias"] as String
                keyPassword = keystoreProperties["keyPassword"] as String
                storeFile = file(keystoreProperties["storeFile"] as String)
                storePassword = keystoreProperties["storePassword"] as String
            }
        }
    }

    buildTypes {
        release {
            signingConfig = if (hasReleaseKey) {
                signingConfigs.getByName("release")
            } else {
                // Upgrades will not install over a build signed with a
                // different key. Fine locally, not fine from CI.
                logger.warn(
                    "SafarSathi: no android/key.properties, so this release " +
                        "APK is signed with debug keys. Installing it over " +
                        "an existing copy will fail with a package conflict."
                )
                signingConfigs.getByName("debug")
            }
        }
    }
}

kotlin {
    compilerOptions {
        jvmTarget = org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17
    }
}

flutter {
    source = "../.."
}
