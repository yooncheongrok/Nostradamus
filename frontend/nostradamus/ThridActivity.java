package com.example.nostradamus;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.widget.TextView;

import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;

public class ThridActivity extends AppCompatActivity {

    private TextView tvDown, tvUp;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_thrid);

        tvDown = findViewById(R.id.tvdown);
        tvUp = findViewById(R.id.tvup);


        FirebaseDatabase database = FirebaseDatabase.getInstance();
        DatabaseReference ref = database.getReference().child("probability");

        ref.addValueEventListener(new ValueEventListener() {
            @Override
            public void onDataChange( DataSnapshot snapshot) {

                String down = snapshot.child("down").getValue().toString();
                String up = snapshot.child("up").getValue().toString();

                tvDown.setText("Down: " + down);
                tvUp.setText("Up: " + up);

            }

            @Override
            public void onCancelled(@NonNull DatabaseError error) {

            }
        });


    }
}

