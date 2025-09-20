import 'package:flutter/material.dart';
import 'dart:convert';
import 'package:web_socket_channel/web_socket_channel.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Parking Monitor',
      theme: ThemeData(
        useMaterial3: true,
        primarySwatch: Colors.blue,
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.blue,
          brightness: Brightness.light,
          surface: Colors.grey[100],
        ),
        cardTheme: CardThemeData(
          elevation: 4,
          shadowColor: Colors.black.withOpacity(0.1),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
          margin: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        ),
        textTheme: const TextTheme(
          titleLarge: TextStyle(fontWeight: FontWeight.bold, fontSize: 24),
          titleMedium: TextStyle(fontWeight: FontWeight.w600, fontSize: 18),
          bodyMedium: TextStyle(fontSize: 14),
        ),
      ),
      darkTheme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.blue,
          brightness: Brightness.dark,
          surface: Colors.grey[900],
        ),
        cardTheme: CardThemeData(
          elevation: 4,
          shadowColor: Colors.black.withOpacity(0.3),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
          margin: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        ),
      ),
      home: const ParkingPage(),
    );
  }
}

class ParkingPage extends StatefulWidget {
  const ParkingPage({super.key});

  @override
  State<ParkingPage> createState() => _ParkingPageState();
}

class _ParkingPageState extends State<ParkingPage> with SingleTickerProviderStateMixin {
  late WebSocketChannel channel;
  Map<String, bool> spotStatus = {};
  bool _isConnected = false;
  late AnimationController _animationController;

  @override
  void initState() {
    super.initState();
    _animationController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 500),
    )..repeat(reverse: true);
    _connectToWebSocket();
  }

  Future<void> _connectToWebSocket() async {
    try {
      channel = WebSocketChannel.connect(Uri.parse('ws://10.0.2.2:8000/ws'));
      channel.stream.listen(
        (message) {
          final data = jsonDecode(message);
          if (data is Map<String, dynamic>) {
            setState(() {
              spotStatus = data.map((key, value) =>
                  MapEntry(key, value is bool ? value : false));
              _isConnected = true;
            });
          }
        },
        onError: (error) => _handleConnectionError(),
        onDone: () => _handleConnectionError(),
      );
    } catch (e) {
      _handleConnectionError();
    }
  }

  void _handleConnectionError() {
    setState(() => _isConnected = false);
    Future.delayed(const Duration(seconds: 5), _connectToWebSocket);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Parking Monitor'),
        centerTitle: true,
        elevation: 0,
        backgroundColor: Theme.of(context).colorScheme.primaryContainer,
        actions: [
          Padding(
            padding: const EdgeInsets.only(right: 8),
            child: FadeTransition(
              opacity: _animationController.drive(CurveTween(curve: Curves.easeInOut)),
              child: IconButton(
                icon: Icon(
                  _isConnected ? Icons.wifi : Icons.wifi_off,
                  color: _isConnected ? Colors.green : Colors.redAccent,
                  size: 28,
                ),
                onPressed: _connectToWebSocket,
              ),
            ),
          ),
        ],
      ),
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [
              Theme.of(context).colorScheme.surface,
              Theme.of(context).colorScheme.surfaceTint.withOpacity(0.1),
            ],
          ),
        ),
        child: _isConnected ? _buildParkingGrid() : _buildConnectionMessage(),
      ),
    );
  }

  Widget _buildConnectionMessage() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          ScaleTransition(
            scale: _animationController.drive(Tween(begin: 0.8, end: 1.0)),
            child: CircularProgressIndicator(
              valueColor: AlwaysStoppedAnimation<Color>(
                Theme.of(context).colorScheme.primary,
              ),
            ),
          ),
          const SizedBox(height: 20),
          Text(
            _isConnected ? 'Loading...' : 'Connecting to server...',
            style: Theme.of(context).textTheme.titleMedium?.copyWith(
                  color: Theme.of(context).colorScheme.onSurface,
                ),
          ),
        ],
      ),
    );
  }

  Widget _buildParkingGrid() {
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        _buildParkingLotSection(1, Colors.blue.shade600),
        const SizedBox(height: 96),
        _buildParkingLotSection(2, Colors.orange.shade600),
      ],
    );
  }

  Widget _buildParkingLotSection(int lotNumber, Color lotColor) {
    final freeSpots = spotStatus.entries
        .where((entry) => entry.key.startsWith('video$lotNumber') && !entry.value)
        .length;
    final totalSpots = 12;
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Container(
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: lotColor.withOpacity(0.1),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Icon(Icons.local_parking, color: lotColor, size: 28),
                ),
                const SizedBox(width: 12),
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Parking Lot $lotNumber',
                      style: Theme.of(context).textTheme.titleMedium?.copyWith(
                            color: lotColor,
                            fontWeight: FontWeight.bold,
                          ),
                    ),
                    Text(
                      '$freeSpots/$totalSpots Free',
                      style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                            color: lotColor.withOpacity(0.8),
                          ),
                    ),
                  ],
                ),
              ],
            ),
            const SizedBox(height: 16),
            // Top row (spots 1-6)
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: List.generate(6, (index) {
                final spotNum = index + 1;
                final isOccupied = spotStatus['video${lotNumber}_zone_$spotNum'] ?? false;
                return Expanded(
                  child: _buildParkingSpot(lotNumber, spotNum, isOccupied, lotColor),
                );
              }),
            ),
            const SizedBox(height: 12),
            // Bottom row (spots 7-12)
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: List.generate(6, (index) {
                final spotNum = index + 7;
                final isOccupied = spotStatus['video${lotNumber}_zone_$spotNum'] ?? false;
                return Expanded(
                  child: _buildParkingSpot(lotNumber, spotNum, isOccupied, lotColor),
                );
              }),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildParkingSpot(int lotNumber, int spotNumber, bool isOccupied, Color color) {
    return AnimatedContainer(
      duration: const Duration(milliseconds: 300),
      curve: Curves.easeInOut,
      margin: const EdgeInsets.symmetric(horizontal: 4, vertical: 4),
      decoration: BoxDecoration(
        color: isOccupied ? Colors.red.shade400 : Colors.green.shade400,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: color.withOpacity(0.3),
          width: 2,
        ),
        boxShadow: [
          BoxShadow(
            color: isOccupied ? Colors.red.withOpacity(0.2) : Colors.green.withOpacity(0.2),
            blurRadius: 8,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          borderRadius: BorderRadius.circular(12),
          onTap: () {
            ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(
                content: Text(
                  'Spot P$spotNumber in Lot $lotNumber is ${isOccupied ? "occupied" : "available"}',
                ),
                backgroundColor: isOccupied ? Colors.red.shade400 : Colors.green.shade400,
                behavior: SnackBarBehavior.floating,
              ),
            );
          },
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(
                isOccupied ? Icons.directions_car : Icons.local_parking,
                color: Colors.white,
                size: 32,
              ),
              const SizedBox(height: 8),
              Text(
                'P$spotNumber',
                style: const TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.bold,
                  fontSize: 16,
                ),
              ),
              const SizedBox(height: 4),
              Text(
                isOccupied ? 'Taken' : 'Free',
                style: TextStyle(
                  color: Colors.white.withOpacity(0.9),
                  fontSize: 12,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  @override
  void dispose() {
    _animationController.dispose();
    channel.sink.close();
    super.dispose();
  }
}