# Issue #7 — Copy, dialer, call, chat

**Size:** M · **Depends on:** #6 · **Blocks:** #9, #43, #50

The diary renders but does nothing. This wires what a tap actually does.

**The workflow this serves:** Yash finds a number, taps it, it lands on the
clipboard, he opens the Android dialer and pastes. Direct dialling stays
available but is not the default. See DECISIONS.md, 2026-09-11.

---

## What each action does

| Action | Mechanism | Logged as |
|---|---|---|
| Tap an entry | `Clipboard.setData` with `phoneE164 ?? phoneRaw` | `copy` |
| `OPEN DIALER` in the toast | `tel:` with an empty path | `dialer` |
| Call | `tel:<number>` | `call` |
| Chat | `https://wa.me/<digits>` | `whatsapp` |
| SMS | `sms:<number>` | `sms` |

Copy needs no package — `Clipboard` is in `flutter/services`. The rest go
through `url_launcher`, and none of them needs a dialer permission.

**Every action is logged**, copy included. Since the dial now happens outside
the app, a copy is the closest thing we observe to a call, and without it the
record of who was reached rots.

## The toast

Ink ground, paper text, sitting above the bottom of the screen. It carries
the word `Copied`, the number on its own line in tabular figures, and an
`OPEN DIALER` button. It holds about three seconds.

Not a `SnackBar`: the design calls for the number to be readable and for the
button to be a stencil mark, and fighting Material's snackbar theming to get
there costs more than drawing it.

## Failure has to be visible

`launchUrl` returns false when no app can handle the scheme, and it throws on
some devices. Both cases show a plain message saying what could not be opened.
A silent no-op on a phone with no dialer is the worst outcome here.

---

## The one thing to verify on a real device

**Does `tel:` with an empty path open the Android dialer, or error?**

If it errors, fall back to an `ACTION_DIAL` intent over a platform channel —
which would be the first platform-specific code in this project — and log a
decision about it. Until someone checks on hardware, the code treats a
failure as "no app can open that" and says so.

---

## Acceptance criteria

All verified. `flutter analyze` clean, 95 tests green.

- [x] Tapping an entry copies `phoneE164` when present, `phoneRaw` otherwise.
- [x] A light haptic fires on copy.
- [x] The toast shows the number, holds about three seconds, then goes away.
- [x] `OPEN DIALER` launches `tel:` with no number, and dismisses the toast.
- [x] Call, chat and SMS launch their schemes with the right number.
- [x] **WhatsApp strips every non-digit** before building the `wa.me` link,
      and refuses an entry with no digits at all.
- [x] Every action writes a CallLog row with the right action string.
- [x] A launch that fails shows a message rather than doing nothing —
      including when the platform throws instead of returning false.
- [x] **A failed action is not logged as if it happened.**
- [ ] The emergency screen gets `heavyImpact` on dial. That is #50.
- [ ] `tel:` with an empty path opens the Android dialer. **Needs hardware.**

## What the goldens caught

Two layout bugs that no widget test would have failed on, both found the
moment the screen was rendered to an image:

1. The copy toast was covered by the New entry button, hiding the
   `OPEN DIALER` action — the entire point of the toast.
2. Before that, the doubled margin rule was painting as a solid grey column,
   because the second hairline was drawn with a `boxShadow` and a shadow
   spreads behind the whole box.

`test/golden_test.dart` now renders the diary in both themes, empty, and
mid-copy. It loads the bundled fonts and the SDK's icon font, so the images
show what a user would see rather than tofu boxes in a substitute typeface.
