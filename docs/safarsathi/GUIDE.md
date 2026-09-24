# SafarSathi — what it is, and how to use it

*A plain-English guide. No technical knowledge assumed.*

---

## Part 1 — What this app is for

### The problem

You are on a road trip through Meghalaya. It is 9pm, raining, and you are an
hour from your homestay on a hill road. You need to call ahead and say you are
running late.

Your phone has one bar and no data.

Everything you would normally reach for is now useless. The booking is in
Gmail — that needs data. The homestay's number is in a WhatsApp message —
that needs data. You could look it up on Google Maps — that needs data. The
one thing that still works on one bar of signal is a **phone call**, and you
cannot make it because you cannot find the number.

### What SafarSathi does

It is a notebook you fill in **before** you leave, while you still have WiFi.
Then it works with no internet at all, forever.

It holds the phone numbers, your route, your packing list, what everyone
spent, and a map you downloaded in advance.

There is no account. No login. No password. Nothing is sent to a server,
because there is no server. Everything lives on your phone and nowhere else.

### The one idea that makes it different

**The app knows the difference between a number you have called and a number
you have merely written down.**

Any number you have not yet dialled carries a small **amber dot**. Once you
call it and it works, you mark it confirmed and the dot goes away — replaced
by a stamp that says CONFIRMED.

That sounds small. It is the whole point.

A hotel number copied off a website six months ago might be disconnected. A
hospital number from an online directory might be a fax line. **You will not
find out at 9pm in the rain.** You find out now, at home, by calling it —
and the app keeps score of which ones you have actually checked.

Before you leave, the app tells you plainly: *"4 numbers not confirmed. Call
before you leave signal."*

### What the app will not do

These are deliberate, not missing features.

- **It will not give you emergency numbers it cannot vouch for.** Every
  emergency number it ships with came from an official government page, and
  the app shows you where each one came from. Numbers we could not verify are
  simply not there. A wrong ambulance number is worse than no number.
- **It will not guess.** If it does not know something, it says so.
- **It will not use the internet while you are travelling.** Only when you
  press a download button, on WiFi, before you go.
- **It will not "refresh".** Pulling down on the list does not fetch anything,
  because there is nothing to fetch. It shows you when you last downloaded
  instead.

---

## Part 2 — How to use it

### The five buttons at the bottom

| Button | What it is |
|---|---|
| **Diary** | All your phone numbers. |
| **Trip** | Your route, and whether you are ready to leave. |
| **Money** | Who paid for what. |
| **SOS** | Emergency numbers. Deliberately plain. |
| **More** | Everything else. |

---

### First time: setting up

**1. Make a trip**

Open the app. Tap **Start a trip**, give it a name and dates.

**2. Add your stops**

**More → Edit the itinerary → Add stop.**

A stop is a place you are staying or passing through. Give it a name, the date
you arrive, and how many nights. If you are only driving through, leave nights
at zero.

Tap **Find on map** to give the stop its real location — this is what makes
weather and offline maps work later.

Drag the handle on the right of a row to reorder stops.

**3. Add your phone numbers**

Three ways, depending on how many you have:

- **One at a time** — Diary → **New entry**.
- **Several at once** — More → **Add several at once**. A list of blank rows.
  Type a name and number, and a new row appears underneath. Good for when you
  have a booking confirmation open.
- **From a spreadsheet** — More → **Import from a sheet**. Pick a CSV or Excel
  file. The app works out which column is which and shows you every row before
  saving anything.

**Say where you are staying at each stop**

A stop can hold several guest houses: options from a sheet, the one you
booked, a backup. The app does not guess which is yours. Choose it in any
of three places:

- **Trip page → Tonight card** → *Where are you staying…? choose yours* (or
  *Change*).
- **The stop's page** (More → Edit the itinerary → tap the stop) →
  **Staying at** → **Choose**.
- **The guest house's own page** → **I am staying here in …**

The one you choose is tonight's bed on the Trip page, the stay in the plan you
send home, and the number that has to be confirmed before the trip reads
ready. The other guest houses stay in the diary. A place you add yourself as
the first stay at a stop counts as your choice straight away. An imported
list never chooses itself.

**4. Call each one and confirm it**

This is the step people skip, and it is the one that matters.

In the Diary, **tap a row** to copy the number, then **Open dialer** and paste.
Call it. If somebody answers:

Long-press the row → **Confirm** → the stamp lands, the amber dot disappears,
and the counter at the top goes down by one.

**5. Download everything**

While on WiFi: **More → Download everything.**

This fetches the road routes, what is along them, the weather forecast, and
the map tiles. It tells you roughly how big it will be before it starts.

*(Maps need a free key from maptiler.com, pasted once into Settings → Map key.
Everything else works without it.)*

**6. Save a backup**

**More → Backup and restore → Save a backup.**

Do this now, not later. Everything is on this one phone. If you lose it, and
there is no backup, it is gone — there is no server holding a copy.

---

### On the road

**To call someone**

Diary → tap the row → the number is copied → **Open dialer** → paste → call.

Or swipe the row right to do the same thing without aiming.

**To get to a place in your diary**

Open the entry. Under **Where it is** there is a small map with the place
pinned, drawn from what you downloaded, so it works with no signal. Tap **How
far am I?** and the phone's GPS puts you on it, with the straight-line
distance. GPS needs no signal either.

**Directions** opens Google Maps with the route from where you are. That
part needs signal, unless you have saved the area as an offline map inside
Google Maps, which can then give driving directions with none.

An entry with no location says so. To add one, tap **Edit**. In Google Maps,
press and hold the place, copy the numbers that appear at the top (like
`25.567739, 91.881081`), and paste them into **Location**. Standing at the
place? Tap **I am here now** instead. A shared `maps.app.goo.gl` link does not
work, because it holds no coordinates.

If your numbers came from a spreadsheet without locations, import the newer
sheet that has **Latitude** and **Longitude** columns. Entries already in the
diary get their location added, and no copies are made.

**In an emergency**

**SOS.** Tap a number and it dials straight away, no copying. This screen is
deliberately plainer than the rest of the app — nothing to read, nothing to
decide.

Under each number it says where it came from. Numbers you added yourself for
this trip appear in their own section below the national ones.

**To see what is coming up on the road**

More → **Getting between stops** → tap a leg.

Shows what is along that road, in order of how far along it is. "Coming up in
12 km" is more useful while driving than a map showing something 200m away
across a gorge.

Anything with an amber dot came from open map data — nobody has checked it.

**To record what you spent**

Money → **+** → amount, what it was for, who paid.

To split with people: Money → **Who is on this trip** → add their names
(first names only — no accounts, nothing is sent anywhere). Then the app works
out the fewest payments needed to settle up at the end.

**To look at the map**

More → **The map.** It draws what you downloaded, with your stops and route
on it. Pan outside the area you fetched and it goes blank — that is the honest
answer, not a bug.

**To check the weather**

More → **Weather.** It shows what you downloaded, **and when you downloaded
it.** Past three days old, it turns amber and says so — because weather in
Meghalaya in October turns over faster than that, and a stale forecast that
looks current is how people pack wrong.

---

### Things worth knowing

**The amber dot means "nobody has checked this".** On a number you typed, it
means you have not called it. On a place from map data, it means it came from
a public map anyone can edit.

**Tapping copies, it does not dial.** Deliberate — it means one tap works the
same way everywhere, and you can paste into WhatsApp just as easily.

**Tags build your packing list.** On a stop, tapping "caves" or "rain" adds the
right things to the checklist. Anything you have changed by hand stays as you
left it.

**Undo a whole import.** More → Import history. If you imported the wrong
sheet, remove all of it in one action.

**The app never asks for permissions it does not need.** One permission:
internet, used only when you press download.

---

## Part 3 — If something goes wrong

**"App not installed as package conflicts with an existing package"**
The new version was signed differently from the one on your phone. Uninstall
the old one first — **but save a backup before you do, because uninstalling
deletes everything.**

**Download says "No internet"**
It could not reach anything at all. Check WiFi is actually working — a WiFi
that is connected but not passing traffic looks identical to a working one.

**Download says "The connection dropped partway through"**
Different problem: it reached the internet and then lost it. Press download
again. Whatever already arrived is kept.

**The map screen says maps are off**
No map key. Settings → Map key, paste a free key from maptiler.com.

**I lost my phone**
Install the app on the new one, then More → Backup and restore → Choose a
backup file. Everything comes back including which numbers you had confirmed.

---

## In one sentence

**Fill it in at home on WiFi, call every number once, and then it works in a
valley with no signal — which is the only place it actually matters.**
