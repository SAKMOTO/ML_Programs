import 'package:flutter/material.dart';

class Exp7Form extends StatefulWidget {
  const Exp7Form({super.key});

  @override
  State<Exp7Form> createState() => _Exp7FormState();
}

class _Exp7FormState extends State<Exp7Form> {
  final _formKey = GlobalKey<FormState>();

  final TextEditingController _dateController =
      TextEditingController();

  String _name = '';
  String _email = '';
  String _password = '';

  String _gender = 'Male';
  String? _selectedCountry;

  bool _agreeToTerms = false;

  final List<String> _countries = [
    'USA',
    'Canada',
    'UK',
    'India',
  ];

  Future<void> _selectDate(BuildContext context) async {
    final DateTime? picked = await showDatePicker(
      context: context,
      initialDate: DateTime.now(),
      firstDate: DateTime(1900),
      lastDate: DateTime(2100),
    );

    if (picked != null) {
      setState(() {
        _dateController.text =
            picked.toLocal().toString().split(' ')[0];
      });
    }
  }

  void _submitForm() {
    if (_formKey.currentState!.validate() &&
        _agreeToTerms) {
      _formKey.currentState!.save();

      showDialog(
        context: context,
        builder: (_) => AlertDialog(
          title: const Text("Form Submitted"),
          content: Text(
            "Thank you, $_name!",
          ),
        ),
      );
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text(
            "Please complete the form and accept terms.",
          ),
        ),
      );
    }
  }

  @override
  void dispose() {
    _dateController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          "User Registration Form",
        ),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Form(
          key: _formKey,
          child: ListView(
            children: [
              // Name Field
              TextFormField(
                decoration: const InputDecoration(
                  labelText: "Name",
                ),
                validator: (value) =>
                    value!.isEmpty
                        ? 'Please enter your name'
                        : null,
                onSaved: (value) =>
                    _name = value!,
              ),

              // Email Field
              TextFormField(
                decoration: const InputDecoration(
                  labelText: "Email",
                ),
                keyboardType:
                    TextInputType.emailAddress,
                validator: (value) =>
                    value!.contains('@')
                        ? null
                        : 'Enter a valid email',
                onSaved: (value) =>
                    _email = value!,
              ),

              // Password Field
              TextFormField(
                decoration: const InputDecoration(
                  labelText: "Password",
                ),
                obscureText: true,
                validator: (value) =>
                    value!.length < 6
                        ? 'Password too short'
                        : null,
                onSaved: (value) =>
                    _password = value!,
              ),

              const SizedBox(height: 20),

              // Gender Radio Buttons
              const Text("Gender"),

              Row(
                children: [
                  Radio<String>(
                    value: 'Male',
                    groupValue: _gender,
                    onChanged: (value) {
                      setState(() {
                        _gender = value!;
                      });
                    },
                  ),
                  const Text("Male"),

                  Radio<String>(
                    value: 'Female',
                    groupValue: _gender,
                    onChanged: (value) {
                      setState(() {
                        _gender = value!;
                      });
                    },
                  ),
                  const Text("Female"),
                ],
              ),

              // Country Dropdown
              DropdownButtonFormField<String>(
                decoration: const InputDecoration(
                  labelText: "Country",
                ),
                initialValue: _selectedCountry,
                items: _countries
                    .map(
                      (country) => DropdownMenuItem(
                        value: country,
                        child: Text(country),
                      ),
                    )
                    .toList(),
                onChanged: (value) {
                  setState(() {
                    _selectedCountry = value;
                  });
                },
                validator: (value) =>
                    value == null
                        ? 'Please select a country'
                        : null,
              ),

              // Date Picker Field
              TextFormField(
                controller: _dateController,
                readOnly: true,
                decoration: const InputDecoration(
                  labelText: "Date of Birth",
                ),
                onTap: () =>
                    _selectDate(context),
                validator: (value) =>
                    value!.isEmpty
                        ? 'Select your date of birth'
                        : null,
              ),

              // Checkbox
              CheckboxListTile(
                title: const Text(
                  "I agree to the Terms & Conditions",
                ),
                value: _agreeToTerms,
                onChanged: (value) {
                  setState(() {
                    _agreeToTerms = value!;
                  });
                },
              ),

              const SizedBox(height: 20),

              // Submit Button
              ElevatedButton(
                onPressed: _submitForm,
                child: const Text("Submit"),
              ),
            ],
          ),
        ),
      ),
    );
  }
}